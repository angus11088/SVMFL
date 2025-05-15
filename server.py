import random
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy  as np
import torch
import math
import tqdm
import copy
import hdbscan
import wandb
import os
from scipy.spatial import distance
# 額外加入基於距離的異常檢測方法
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.distance import cosine
from scipy.stats import trim_mean
from scipy.stats import multivariate_normal
import seaborn as sns
from sklearn import svm
from sklearn.cluster import KMeans, DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from models import model_eval, cal_metrics
from utils import weighted_avg_params, weighted_avg, weighted_avg_with_momentum_forType, weighted_avg_with_momentum_ACG
from torchmetrics.functional import pairwise_cosine_similarity
from XGBoostClassifier import XGBoostClassifier
from build import SERVER_REGISTRY

# 步驟 1：初始化與設置環境
os.environ["OMP_NUM_THREADS"] = "1"
# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
# FedAwS cosine similarity margin
margin = 0

def filter_outlier_clients(global_model, client_models, client_weights, method='zscore', threshold=2.5):
    deltas = []
    global_device = next(global_model.parameters()).device
    for model in client_models:
        squared_diff = 0.0
        for p_global, p_client in zip(global_model.parameters(), model.parameters()):
            p_client = p_client.to(global_device)
            squared_diff += torch.norm(p_global.data - p_client.data, p=2).item() ** 2
        deltas.append(squared_diff ** 0.5)

    deltas = np.array(deltas)

    if method == 'zscore':
        mean = deltas.mean()
        std = deltas.std()
        keep_mask = np.abs(deltas - mean) <= threshold * std if std != 0 else np.ones_like(deltas, dtype=bool)
    elif method == 'iqr':
        q1 = np.percentile(deltas, 25)
        q3 = np.percentile(deltas, 75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        keep_mask = (deltas >= lower_bound) & (deltas <= upper_bound)
    else:
        raise ValueError("未知過濾方式")

    all_filtered = not np.any(keep_mask)
    if all_filtered:
        print("⚠️ 所有客戶端皆被判定為異常，將跳過過濾，使用所有 client 進行聚合")
        return client_models, client_weights, True  # 保留 True 作為 fallback indicator


    filtered_models = [m for m, keep in zip(client_models, keep_mask) if keep]
    filtered_weights = [w for w, keep in zip(client_weights, keep_mask) if keep]
    print(f"🧹 剔除 {len(client_models) - len(filtered_models)} 個異常 client 更新")
    return filtered_models, filtered_weights, False

def federated_learning(args: object, train_clients: list[object], test_clients: list[object], global_model: torch.nn.Module) -> None:
    """
    Main loop for federated learning.

    Arguments:
        args (argparse.Namespace): parsed argument object.
        train_clients (list[Client]): training clients.
        test_clients (list[Client]): test / validation clients.
        global_model (torch.nn.Module): pytorch model (global model on the server).
    """
    #  # ✅ 初始化 momentum 結構（只做一次）
    # if not hasattr(global_model, 'logits_momentum'):
    #     global_model.logits_momentum = {}
    #     global_model.logits_delta = {}
    # determine how many clients are updated per global round
    num_train_client  = len(train_clients)
    if args.client_C < 1.0: # proportion
        num_update_client = min(max(math.ceil(args.client_C * num_train_client), 1), num_train_client) # number of clients to update per round
    else: # client_C itself is num_update_client
        num_update_client = min(args.client_C, num_train_client)

    # global optimizer
    global_model.to(device)
    global_optim = args.global_optim(global_model.parameters(), lr = args.global_lr, amsgrad = args.amsgrad)
    logits_optim = args.logits_optim(global_model.logits.parameters(), lr = args.logits_lr, eps = 1e-5)

    # 步驟 2：資料載入與處理
    # train-valid-test split on server level
    #print("server.py => Server 把使用者資料合成成一個資料流變數")
    #print("server.py => train_clients長度",len(train_clients))
    #print("server.py => test_clients長度",len(test_clients))
    global_train_dataset = torch.utils.data.ConcatDataset([c.dataset for c in train_clients])
    global_test_dataset  = torch.utils.data.ConcatDataset([c.dataset for c in test_clients ])
    global_train_loader  = torch.utils.data.DataLoader(global_train_dataset, batch_size = args.global_bs, shuffle = False, pin_memory = True)
    global_test_loader   = torch.utils.data.DataLoader(global_test_dataset , batch_size = args.global_bs, shuffle = False, pin_memory = True)
    
    # 步驟 3：計算全局模型性能
    # performance before training
    print("計算global model性能")
    wandb_log = {}
    model_eval(global_model, global_train_loader, wandb_log, 'train/')
    model_eval(global_model, global_test_loader , wandb_log, 'test/' )
    wandb.log(wandb_log)
    
    # for MOON
    previous_features = None
    
    last_loss = 5.0
    history_acc = []

    # global round loop
    print()
    # 步驟 4：選擇參與更新的客戶端
    for current_global_epoch in tqdm.tqdm(range(args.global_epoch)):
         # ✅ Server 端 logits 預更新（lookahead），先行推進 global_model 的 logits 權重
        # apply_server_side_lookahead(global_model)
        # 在每輪訓練後判斷是否要終止訓練

        if len(history_acc) >= 20:
            history_acc.pop(0)
        epoch_train_times = 1

        # select clients which are updated in this round
        update_clients = np.random.choice(train_clients, num_update_client, replace = False)

        client_weights = [c.num_sample for c in update_clients]
        client_models  = [copy.deepcopy(global_model) for c in update_clients]

        continue_training = True
        tmp_global_model = copy.deepcopy(global_model)

        # 步驟 5：訓練客戶端
        # training
        if args.switch_FL == "FedEFC":
            # ✅ FedEFC 對應的原始 for 迴圈
            for client, client_model in zip(update_clients, client_models):
                previous_features = client.local_train(client_model, global_model, previous_features)

        elif args.switch_FL == "FedGMMDBACG":
            # ✅ FedGMMDBACG 對應的新版本，會回傳 updated_model
            for i, (client, client_model) in enumerate(zip(update_clients, client_models)):
                updated_model, previous_features = client.local_train(client_model, global_model, previous_features)
                client_models[i] = updated_model  # ✅ overwrite 原本的 client_model
        else:
            # ✅ 其他的 for 迴圈，這裡不需要改動
            for client, client_model in zip(update_clients, client_models):
                previous_features = client.local_train(client_model, global_model, previous_features)

        # 步驟 6：模擬惡意客戶端
        # 根據設定，隨機選擇一定數量的客戶端（malicious_indices），這些客戶端將進行惡意攻擊。
        # 根據所選的攻擊強度（Weak 或 Strong），對選中的客戶端模型進行擾動處理。這些擾動會影響模型的參數，並模擬現實中惡意客戶端的攻擊行為。
# ----------------實驗設計添加惡意等級，分最惡意、惡意、正常等級-------------------
        # 1. 隨機選擇 2～3 個惡意 client index
        num_clients = len(client_models)
        if num_clients <= 10:
            num_malicious = 1 #若client數量小於10，則隨機選擇1個client，client為8、惡意客戶端為2或以上的時候，我的訓練會崩潰
        else:
            num_malicious = min(3, max(2, num_clients // 4))

        malicious_indices = random.sample(range(num_clients), num_malicious)
        # -----------------------------------
        # ** 模擬惡意客戶端-多個的 - TurboSVM的** 較為平凡的惡意 (比較偵測不出來)
        if args.malicious == 'Weak':
            for idx in malicious_indices:
                malicious_model = client_models[idx]
                with torch.no_grad():
                    for name, p in malicious_model.named_parameters():
                        # 對所有層進行擾動（若要保守，也可跳過logits層）
                        direction = torch.sign(torch.randn_like(p.data))
                        scale = random.uniform(15, 40)
                        perturbation = scale * direction * torch.randn_like(p.data)
                        p.data += perturbation

                        # 數值穩定化處理（防止爆炸或NaN）
                        torch.nan_to_num_(p.data, nan=1e-5, posinf=1e2, neginf=-1e2)

                    # 額外針對logits層做限制（可選）
                    if hasattr(malicious_model, 'logits'):
                        if hasattr(malicious_model.logits, 'weight'):
                            malicious_model.logits.weight.data = torch.clamp(malicious_model.logits.weight.data, -20, 20)
                        if hasattr(malicious_model.logits, 'bias'):
                            malicious_model.logits.bias.data = torch.clamp(malicious_model.logits.bias.data, -20, 20)
            print(f"✅ 本輪惡意客戶端 indices: {malicious_indices}")
        # ** 模擬惡意客戶端-多個的 - TurboSVM的**
        # -----------------------------------

        #------------------------------------
        #強化版惡意(比較偵測的了差別)
        if args.malicious == 'Strong':
            for idx in malicious_indices:
                malicious_model = client_models[idx]
                with torch.no_grad():
                    for name, p in malicious_model.named_parameters():
                        # 💥 更極端的擾動：強度大、方向固定、隨機波動加劇
                        direction = torch.sign(torch.ones_like(p.data))  # 固定正方向（可改為 -1 負方向）
                        scale = random.uniform(50, 150)  # 擾動強度加大
                        perturbation = scale * direction * torch.randn_like(p.data) * 2.0  # 擴大隨機性
                        p.data += perturbation

                        # 🛡 數值穩定化（仍保留以避免 NaN 爆炸）
                        torch.nan_to_num_(p.data, nan=1e-5, posinf=1e3, neginf=-1e3)

                    # ❌ 移除 logits clamp（放寬惡意破壞能力）
                    # 若希望更具破壞性，也可針對 logits 設為固定值（例如極大正負值）
                    if hasattr(malicious_model, 'logits'):
                        if hasattr(malicious_model.logits, 'weight'):
                            malicious_model.logits.weight.data += torch.randn_like(malicious_model.logits.weight.data) * 100
                        if hasattr(malicious_model.logits, 'bias'):
                            malicious_model.logits.bias.data += torch.randn_like(malicious_model.logits.bias.data) * 50
            
            print(f"⚠️ 惡意攻擊已強化，目標客戶端 indices: {malicious_indices}")
        #強化版惡意
        #------------------------------------
        
        if args.switch_FL == 'TurboSVM':
            eval(args.fed_agg)(global_model, client_models, client_weights, # basic FL parameters
                            global_optim, # for FedOpt (FedAdam and FedAMS)
                            logits_optim, # for FedAwS and TurboSVM
                            current_global_epoch, args.global_epoch, args.class_C, args.base_agg, args.agg_svc, args.spreadout)
            # stability
            for p in global_model.parameters():
                torch.nan_to_num_(p.data, nan=1e-5, posinf=1e-5, neginf=1e-5)

        # 步驟 7：過濾異常客戶端的更新
        # 使用 filter_outlier_clients 函數檢測並過濾那些更新與全局模型過度偏離的客戶端。
        # 這個過程會根據 zscore 或 IQR 方法來確定哪些更新是異常的，這有助於提高聯邦學習的穩定性。
        while continue_training and (args.switch_FL == 'FedEFC' or args.switch_FL == 'FedGMMDBACG'):
            # 全局模型聚合
            # ✅ 在聚合前移除異常更新(梯度爆炸的class去除)
            client_models, client_weights, all_filtered = filter_outlier_clients(
                global_model, client_models, client_weights, method='zscore', threshold=2.5
            )

            # 步驟 8：模型聚合與更新
            # 根據不同的聯邦學習方法（FedEFC、FedGMMDBACG、TurboSVM 等），進行全局模型的聚合。
            eval(args.fed_agg)(
                global_model, client_models, client_weights,  # 基本聯邦學習參數
                global_optim,  # 用於 FedOpt（FedAdam 和 FedAMS）
                logits_optim,  # 用於 FedAwS 和 TurboSVM
                current_global_epoch, args.global_epoch, args.class_C, args.base_agg, args.agg_svc, args.spreadout,  # 用於 TurboSVM
                args.cluster_method, args.num_clusters, args.dbscan_eps, args.dbscan_num_sample, args.client_epoch,  # 用於 FedGMMDBACG
            )
            # 步驟 9：穩定性處理
            # 穩定性處理，使用 torch.nan_to_num_ 來將模型參數中的 NaN 或無窮大值替換為合理的數值。
            for p in global_model.parameters():
                torch.nan_to_num_(p.data, nan=1e-5, posinf=1e-5, neginf=1e-5)

            global_train_loader = torch.utils.data.DataLoader(
                global_train_dataset, batch_size=args.global_bs, shuffle=False, pin_memory=True
            )
            labels, preds = model_eval(global_model, global_train_loader, wandb_log, 'train/', True)
            acc = accuracy_score(preds.argmax(axis=1), labels)
            history_acc.append(acc)

            # 結束訓練
            break

        # 步驟 10：性能評估
        # performance metrics
        global_train_dataset = torch.utils.data.ConcatDataset([c.dataset for c in update_clients])
        global_train_loader  = torch.utils.data.DataLoader(global_train_dataset, batch_size = args.global_bs, shuffle = False, pin_memory = True)
        wandb_log = {}
        model_eval(global_model, global_train_loader, wandb_log, 'train/')
        model_eval(global_model, global_test_loader , wandb_log, 'test/' )
        wandb_log['epoch_train_times'] = epoch_train_times
        wandb.log(wandb_log)
       
    # global_model.to('cpu')
    # wandb.finish()

def server_eval(clients: list[object], wandb_log: dict[str, float], metric_prefix: str) -> None:
    """
    (Obsolete.) Evaluate model performance globally by letting each client conduct inference locally and then collecting all inferences and calculating metrics.

    Arguments:
        clients (list[Client]): list of clients.
        wandb_log (dict[str, float]): wandb log dictionary, with metric name as key and metric value as value.
        metric_prefix (str): prefix for metric name.
    """

    labels = []
    preds  = []
    for c in clients:
        l, p = c.local_eval()
        labels.append(l)
        preds .append(p)
    labels = torch.cat(labels)
    preds  = torch.cat(preds )
    cal_metrics(labels, preds, wandb_log, metric_prefix)    

def FedAvg(global_model: torch.nn.Module, client_models: list[torch.nn.Module], client_weights: list[int], *_) -> None:
    """
    Federated learning algorithm FedAvg.

    Arguments:
        global_model (torch.nn.Module): pytorch model (global model).
        client_models (list[torch.nn.Module]): pytorch models (client models).
        client_weights (list[int]): number of samples per client.
    """

    client_params  = [m.state_dict() for m in client_models]
    new_global_params = weighted_avg_params(params = client_params, weights = client_weights)
    global_model.load_state_dict(new_global_params)

def FedOpt(global_model: torch.nn.Module, client_models: list[torch.nn.Module], client_weights: list[int], global_optim: torch.optim, *_) -> None:
    """
    Federated learning algorithm FedOpt. Depending on the choice of optimizer, it can be deviated into different variates like FedAdam and FedAMS.

    Arguments:
        global_model (torch.nn.Module): pytorch model (global model).
        client_models (list[torch.nn.Module]): pytorch models (client models).
        client_weights (list[int]): number of samples per client.
        global_optim (torch.optim): pytorch optimizer for global model.
    """

    client_params  = [m.state_dict() for m in client_models]
    new_global_params = weighted_avg_params(params = client_params, weights = client_weights)
    
    # pseudo-gradient
    global_model.train()
    for p_name, p in global_model.named_parameters():
        if p.requires_grad:
            p.grad = global_model.state_dict()[p_name] - new_global_params[p_name].to(p.device)
    
    # apply optimizer
    global_optim.step()
    global_optim.zero_grad()

def FedAwS(global_model: torch.nn.Module, 
           client_models: list[torch.nn.Module], 
           client_weights: list[int], 
           global_optim: torch.optim, 
           logits_optim: torch.optim, 
           *_) -> None:
    """
    Federated learning algorithm FedAwS.

    Arguments:
        global_model (torch.nn.Module): pytorch model (global model).
        client_models (list[torch.nn.Module]): pytorch models (client models).
        client_weights (list[int]): number of samples per client.
        global_optim (torch.optim): (useless) pytorch optimizer for global model.
        logits_optim (torch.optim): pytorch optimizer for logit layer of global model.
    """

    FedAvg(global_model, client_models, client_weights)
    global_model.train()
    
    # spreadout regularizer
    wb = torch.cat((global_model.logits.weight, global_model.logits.bias.view(-1, 1)), axis = 1)
    cos_sim_mat = pairwise_cosine_similarity(wb)
    cos_sim_mat = (cos_sim_mat > margin) * cos_sim_mat
    loss = cos_sim_mat.sum() / 2
    loss.backward()
    
    # apply optimizer
    logits_optim.step()
    logits_optim.zero_grad()

# TurboSVM-FL
def TurboSVM(global_model: torch.nn.Module, 
         client_models: list[torch.nn.Module], 
         client_weights: list[int], 
         global_optim: torch.optim, 
         logits_optim: torch.optim, 
         current_global_epoch: int, 
         num_global_epoch: int, 
         class_C: int | float, 
         base_agg: str, 
         agg_svc: bool, 
         spreadout: bool,
         current_client_ids: list[int] = None,
         *_
         ) -> None:
    """
    Federated learning algorithm TurboSVM-FL.

    Arguments:
        global_model (torch.nn.Module): pytorch model (global model).
        client_models (list[torch.nn.Module]): pytorch models (client models).
        client_weights (list[int]): number of samples per client.
        global_optim (torch.optim): pytorch optimizer for global model.
        logits_optim (torch.optim): pytorch optimizer for logit layer of global model.
        current_global_epoch (int): current global aggregation round.
        num_global_epoch (int): total number of global aggregation rounds.
        class_C (int|float): number of classes used to train SVM. Can be int (absolute value) or float (proportion).
        base_agg (str): FL aggregation algorithm for encoder (all layers of global model but logit layer).
        agg_svc (bool): whether to aggregate only client models that form support vectors or all client models.
        spreadout (bool): whether to apply max-margin spread-out regularization or not.
    """
    #print("進入TurboSVM Function")
    #print(f'client_models : {len(client_models)}----------------')
    #print(f'client_weights : {len(client_weights)}----------------')
    # aggregate client models
    assert(base_agg == 'FedAvg' or base_agg == 'FedOpt')
    eval(base_agg)(global_model, client_models, client_weights, global_optim)
    
    # randomly select classes, whose embeddings will be updated using TurboSVM
    num_class = global_model.logits.weight.shape[0]
    if class_C <= 1.0: # proportion
        num_update_class = min(max(math.floor(class_C * num_class), 2), num_class)
    else: # class_C itself is num_update_class
        num_update_class = min(class_C, num_class)
    classes = np.random.choice(num_class, num_update_class, replace = False).tolist()
    classes.sort() # must be sorted when more than 2 classes, as svm coefs are sorted based on class id
    # print("number of SVC class:", num_update_class)
    # print(classes)
    # class embeddings, class labels, and client weights
    x, y, w = [], [], []
    for m, cw in zip(client_models, client_weights):
        wb = torch.cat((m.logits.weight[classes], m.logits.bias[classes].view(-1, 1)), axis = 1).detach()
        x.append(wb)
        y += classes
        w += [cw] * num_update_class
    x = torch.cat(x)
    assert(len(x) == len(y) == len(w)) # x, y, w should have same length
    # print(f'X,Y,W LEN : {len(x)}----------------')
    # print(f'SVM LEN : {num_class}----------------')
    # fit SVM
    C = (num_global_epoch - current_global_epoch) / num_global_epoch # decreasing C successively --> increasing number of support vectors successively
    # print(f'C: {C} (decreasing C;increasing number of support vectors) ----------------')
    clf = svm.SVC(kernel = 'linear', max_iter = 50, tol = 1e-3, C = C)
    clf.fit(x, y)

    # === 印出未提供任何 support vector 的 clients ===
    client_contrib = num_update_class
    support_indices = set(clf.support_)
    removed_local_ids = []

    for i in range(len(client_models)):
        start = i * client_contrib
        end = start + client_contrib
        if not any(idx in support_indices for idx in range(start, end)):
            removed_local_ids.append(i)

    if removed_local_ids:
        if current_client_ids is not None:
            removed_global_ids = [current_client_ids[i] for i in removed_local_ids]
            print(f"🛑 TurboSVM 被 SVM 篩除的 clients (global indices): {removed_global_ids}")
        else:
            print(f"🛑 TurboSVM 被 SVM 篩除的 clients (local indices): {removed_local_ids}")
    # collect support vectors and their weights
    SVCs = {class_id : [] for class_id in classes}
    ws   = {class_id : [] for class_id in classes}
    for svc_id in clf.support_:
        svc   = x[svc_id]
        svc_w = w[svc_id]
        class_id = y[svc_id]
        
        SVCs[class_id].append(svc)
        ws  [class_id].append(svc_w)
    # aggregate support vectors and update logit weight as well as logit bias
    if agg_svc:
        with torch.no_grad():
            for class_id in classes:
                new_wb = weighted_avg(SVCs[class_id], ws[class_id],current_global_epoch)
                global_model.logits.weight[class_id] = new_wb[:-1]
                global_model.logits.bias  [class_id] = new_wb[-1]

    # apply spreadout regularization
    if spreadout:
        loss = 0
        h_id = 0 # id for hyperplane
        for class_a_idx in range(num_update_class):
            class_a = classes[class_a_idx]
            w_a = torch.cat([global_model.logits.weight[class_a], global_model.logits.bias[class_a].view(-1)])
            
            for class_b_idx in range(class_a_idx + 1, num_update_class):
                class_b = classes[class_b_idx]
                w_b = torch.cat([global_model.logits.weight[class_b], global_model.logits.bias[class_b].view(-1)])

                hyperplane_ab = torch.tensor(clf.coef_[h_id]).float().to(device)
                similarity = torch.exp(- ((torch.dot(w_a - w_b, hyperplane_ab) / hyperplane_ab.norm()) ** 2) / 2)
                loss += similarity

                h_id += 1

        loss.backward()
        logits_optim.step()
        logits_optim.zero_grad()
    
def FedEFC(global_model: torch.nn.Module, 
           client_models: list[torch.nn.Module], 
           client_weights: list[int], 
           global_optim: torch.optim, 
           logits_optim: torch.optim, 
           current_global_epoch: int, 
           num_global_epoch: int, 
           class_C: int | float, 
           base_agg: str, 
           agg_svc: bool, 
           spreadout: bool,
           cluster_method: str,
           num_clusters: int,
           dbscan_eps: int | float,
           dbscan_num_sample: int,
           # 新增高斯混合模型參數
           gmm_num_clusters: int = 2,  # 默認聚類數量為 2
           gmm_covariance_type: str = 'full',  # 默認設為 'full'
           gmm_tol: float = 1e-3,  # 默認容忍誤差為 1e-3
           gmm_max_iter: int = 100,  # 默認最大迭代次數為 100
           covariance_type: str = 'full',  # 默認設為 'full'
           tol: float = 1e-3,  # 默認容忍誤差為 1e-3
           max_iter: int = 100,  # 默認最大迭代次數為 100
           random_state: int = 0  # 默認隨機種子
           ) -> None:
    # 這裡是函數體，會根據具體需求執行相應的操作

    """
    Federated learning algorithm FedEFC.

    Arguments:
        global_model (torch.nn.Module): pytorch model (global model).
        client_models (list[torch.nn.Module]): pytorch models (client models).
        client_weights (list[int]): number of samples per client.
        clustering_num :預設K值
    """

    assert(base_agg == 'FedAvg' or base_agg == 'FedOpt')
    eval(base_agg)(global_model, client_models, client_weights, global_optim)
    
    # randomly select classes, whose embeddings will be updated using TurboSVM
    num_class = global_model.logits.weight.shape[0]
    if class_C <= 1.0: # proportion
        num_update_class = min(max(math.floor(class_C * num_class), 2), num_class)
    else: # class_C itself is num_update_class
        num_update_class = min(class_C, num_class)
    classes = np.random.choice(num_class, num_update_class, replace = False).tolist()
    classes.sort() # must be sorted when more than 2 classes, as svm coefs are sorted based on class id
    # print("number of SVC class:", num_update_class)
    # print(classes)
    # class embeddings, class labels, and client weights
    
    x, y, w = [], [], []
    for m, cw in zip(client_models, client_weights):
        wb = torch.cat((m.logits.weight[classes], m.logits.bias[classes].view(-1, 1)), axis = 1).detach()
        x.append(wb)
        y += classes
        w += [cw] * num_update_class
    x = torch.cat(x)
    assert(len(x) == len(y) == len(w)) # x, y, w should have same length
    # 確保 cluster_method 參數有正確傳遞
    print(f"DEBUG: cluster_method = {cluster_method}")

    
# 寫個高斯混合分布
# 新增惡意使用者的資料集，查看是否會影響到模型的訓練、預測、時間速度準確度等
    # 添加 'GaussianMixture' 的情況處理
    match cluster_method:
        case 'KMeans':
            # 對數據進行標準化處理
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            # 使用K-Means進行聚類
            kmeans = KMeans(n_clusters=num_clusters, max_iter=50, tol=1e-3)
            kmeans.fit(x_scaled)
            labels = kmeans.labels_

        case 'KMedoids':
            # 對數據進行標準化處理
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            # 使用K-Medoids進行聚類
            kmedoids = KMedoids(n_clusters=num_clusters, max_iter=50, method='pam', random_state=0)
            kmedoids.fit(x_scaled)
            labels = kmedoids.labels_

        case 'GaussianMixture':
            print("Running GaussianMixture...")
            # 使用 GaussianMixture 聚類
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            
            gmm = GaussianMixture(
                n_components=gmm_num_clusters,
                covariance_type=gmm_covariance_type,
                tol=gmm_tol,
                max_iter=gmm_max_iter,
                random_state=random_state
            )
            gmm.fit(x_scaled)
            # 獲得每個數據點對每個群集的隸屬概率
            # 獲得每個群集的均值和協方差
            means = gmm.means_
            covariances = gmm.covariances_
            # 獲得每個點對每個群集的隸屬度
            probs = gmm.predict_proba(x_scaled)
            # 計算每個群集的集中度（協方差的行列式或對角線元素之和）
            concentration = np.array([np.sum(np.diagonal(cov)) for cov in covariances])
            # 設置閾值篩選數據
            threshold = 0.5  # 設置隸屬度閾值
            # 獲取每個點對其最有可能屬於的群集的隸屬度
            max_prob = np.max(probs, axis=1)
            # 選擇隸屬度高於閾值的樣本
            valid_samples = max_prob >= threshold
            # 篩選出有效樣本和它們的群集標籤
            x_filtered = x_scaled[valid_samples]
            filtered_probs = probs[valid_samples]
            # 獲得每個有效樣本的標籤
            labels = gmm.predict(x_filtered)
            # 根據集中度和隸屬度篩選出好的群集
            # 假設我們選擇協方差較小且隸屬度較高的群集
            good_clusters = concentration < np.percentile(concentration, 50)  # 假設我們選擇協方差較小的群集
            good_cluster_labels = []

            for i, label in enumerate(labels):
                if good_clusters[label]:
                    good_cluster_labels.append(label)

            # 輸出選擇的“好的”群集的標籤
            print(f"Good clusters labels: {good_cluster_labels}")

        case 'HDBSCAN':
            # 對數據進行標準化處理
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            # 使用HDBSCAN進行聚類
            hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=dbscan_num_sample)
            labels = hdbscan_clusterer.fit_predict(x_scaled)

        case 'None':
            # 無聚類方法，將全部數據視為一個簇
            labels = np.zeros(len(x), dtype=int)

        case 'GaussianMixtureDBSCAN':
            print("Running GMM-only anomaly detection...")

            scaler = StandardScaler()
            x_raw = np.array([np.concatenate([p.data.cpu().numpy().flatten() for p in m.parameters()]) for m in client_models])
            x_scaled = scaler.fit_transform(x_raw)

            pca_dim = min(10, x_scaled.shape[0], x_scaled.shape[1])
            if x_scaled.shape[1] > pca_dim:
                pca = PCA(n_components=pca_dim)
                x_scaled = pca.fit_transform(x_scaled)

            valid_clients = np.full(len(x_scaled), True)  # 預設全部 valid
            labels = np.full(len(x_scaled), -1)

            # try:
            gmm = GaussianMixture(
                n_components=min(gmm_num_clusters, len(x_scaled) // 2),
                covariance_type=gmm_covariance_type,
                tol=gmm_tol,
                max_iter=gmm_max_iter,
                random_state=random_state
            )
            gmm.fit(x_scaled)
            gmm_labels = gmm.predict(x_scaled)
            probs = gmm.predict_proba(x_scaled)
            max_prob = np.max(probs, axis=1)
            print("📊 GMM label 分佈:", np.bincount(gmm_labels))
            
            # 🧠 找出最大群集
            counts = np.bincount(gmm_labels)
            majority_cluster = np.argmax(counts)

            # ✅ 條件一：屬於最大群集
            in_majority = (gmm_labels == majority_cluster)

            # ✅ 條件二：機率在主群集的下分位門檻以上
            # ✅ 條件二：機率在主群集的中位數以上（更嚴格）
            prob_threshold = np.median(max_prob[in_majority])
            prob_pass = (max_prob >= prob_threshold)


            # ✅ 條件三：Mahalanobis距離 Z-score < 90%
            def mahalanobis_dist(x, mean, cov_inv):
                return distance.mahalanobis(x, mean, cov_inv)

            distances = np.array([
                mahalanobis_dist(x_scaled[i], gmm.means_[gmm_labels[i]], np.linalg.inv(gmm.covariances_[gmm_labels[i]]))
                for i in range(len(x_scaled))
            ])
            # z_scores = (distances - np.mean(distances)) / np.std(distances)
            # z_pass = np.abs(z_scores) < 1  # Z-score 小於 1（更合理界定）
            z_pass = distances < np.percentile(distances, 90)  # 保留前 90% 接近的客戶端

            # 來自 GMM 分群的分析結果。
            # 有效 client 要同時滿足：
            # 1️⃣ 屬於最大群集 (in_majority)
            # 2️⃣ 在該群集的分群機率高於 25% 分位 (prob_pass)
            # 3️⃣ Mahalanobis 距離的 Z-score < 1 (z_pass)
            # 🎯 GMM 最終 valid clients
            gmm_valid = in_majority & prob_pass & z_pass
             # 額外加入 weight norm 檢查（強化不一致異常偵測）
            weight_norms = np.linalg.norm(x_scaled, axis=1)
            norm_mean, norm_std = np.mean(weight_norms), np.std(weight_norms)
            norm_z = (weight_norms - norm_mean) / norm_std
            norm_pass = (np.abs(norm_z) < 2.5)

            # 訓練 Isolation Forest 模型
            iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=random_state)
            iso_forest.fit(x_scaled)
            iso_labels = iso_forest.predict(x_scaled)
            iso_pass = (iso_labels == 1)  # 標記為正常的 client

            # 訓練 LOF (Local Outlier Factor)
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            lof_labels = lof.fit_predict(x_scaled)
            lof_pass = (lof_labels == 1)  # 標記為正常的 client 

            # 進一步的 DBSCAN 檢測
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_num_sample)
            dbscan_labels = dbscan.fit_predict(x_scaled)

            # DBSCAN 噪聲標籤 -1 代表噪聲
            dbscan_valid = (dbscan_labels != -1)  # 只保留非噪聲點
            dbscan_valid_indices = np.where(dbscan_valid)[0].tolist()
            print(f"\n🐝 DBSCAN 有效 clients indices: {dbscan_valid_indices}")
            print("------------------------------------------------------------")

            # 最後的有效 client 條件：所有條件滿足才是有效
            if(dbscan_valid.sum() <= class_C/2):
                valid_clients = gmm_valid & norm_pass & iso_pass & lof_pass
            else:
                valid_clients = gmm_valid & dbscan_valid & norm_pass & iso_pass & lof_pass

            # 後續處理
            filtered_client_models = [m for i, m in enumerate(client_models) if valid_clients[i]]
            filtered_client_weights = [w for i, w in enumerate(client_weights) if valid_clients[i]]

            if len(filtered_client_models) > 0:
                print("🔁 使用 valid clients 更新 global model")
                eval(base_agg)(global_model, filtered_client_models, filtered_client_weights, global_optim)
            else:
                print("⚠️ 沒有 valid clients，改採用 GMM 最大群集進行 fallback 聚合")

                # fallback：選擇 GMM 最大群集作為 valid_clients
                counts = np.bincount(gmm_labels)
                majority_cluster = np.argmax(counts)
                fallback_valid_clients = (gmm_labels == majority_cluster)

                # 使用 fallback valid clients
                filtered_client_models = [m for i, m in enumerate(client_models) if fallback_valid_clients[i]]
                filtered_client_weights = [w for i, w in enumerate(client_weights) if fallback_valid_clients[i]]
                
                if len(filtered_client_models) > 0:
                    print("🔁 使用 fallback clients 更新 global model")
                    eval(base_agg)(global_model, filtered_client_models, filtered_client_weights, global_optim)
                else:
                    print("⛔ fallback 聚合也失敗，跳過這一輪")

            print(f"✅ Valid clients: {np.sum(valid_clients)} / {len(client_models)}")
            print(f"✅ GMM valid: {np.sum(gmm_valid)} / {len(gmm_valid)}")
            print(f"✅ Norm pass: {np.sum(norm_pass)} / {len(norm_pass)}")
            print(f"✅ ISO pass: {np.sum(iso_pass)} / {len(iso_pass)}")
            print(f"✅ LOF pass: {np.sum(lof_pass)} / {len(lof_pass)}")
            print(f"✅ DBSCAN pass: {np.sum(dbscan_valid)} / {len(dbscan_valid)}")

        case 'GMMDBSCAN':
            print("Running GMM + DBSCAN + Others anomaly detection...")

            # --- 萃取特徵 ---
            scaler = StandardScaler()
            x_raw = np.array([
                np.concatenate([p.data.cpu().numpy().flatten() for p in m.parameters()])
                for m in client_models
            ])
            x_scaled = scaler.fit_transform(x_raw)

            # --- PCA 降維 ---
            pca_dim = min(10, x_scaled.shape[0], x_scaled.shape[1])
            if x_scaled.shape[1] > pca_dim:
                pca = PCA(n_components=pca_dim)
                x_scaled = pca.fit_transform(x_scaled)

            # --- GMM ---
            gmm = GaussianMixture(
                n_components=min(gmm_num_clusters, len(x_scaled) // 2),
                covariance_type=gmm_covariance_type,
                tol=gmm_tol,
                max_iter=gmm_max_iter,
                random_state=random_state
            )
            gmm.fit(x_scaled)
            gmm_labels = gmm.predict(x_scaled)
            probs = gmm.predict_proba(x_scaled)
            max_prob = np.max(probs, axis=1)
            counts = np.bincount(gmm_labels)
            majority_cluster = np.argmax(counts)
            in_majority = (gmm_labels == majority_cluster)
            prob_threshold = np.median(max_prob[in_majority])
            prob_pass = (max_prob >= prob_threshold)

            def mahalanobis_dist(x, mean, cov_inv):
                return distance.mahalanobis(x, mean, cov_inv)

            distances = np.array([
                mahalanobis_dist(x_scaled[i], gmm.means_[gmm_labels[i]], np.linalg.inv(gmm.covariances_[gmm_labels[i]]))
                for i in range(len(x_scaled))
            ])
            z_pass = distances < np.percentile(distances, 90)

            gmm_valid = in_majority & prob_pass & z_pass

            # --- Norm ---
            weight_norms = np.linalg.norm(x_scaled, axis=1)
            norm_z = (weight_norms - np.mean(weight_norms)) / np.std(weight_norms)
            norm_pass = np.abs(norm_z) < 2.5

            # --- Isolation Forest ---
            iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=random_state)
            iso_labels = iso_forest.fit_predict(x_scaled)
            iso_pass = (iso_labels == 1)

            # --- LOF ---
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            lof_labels = lof.fit_predict(x_scaled)
            lof_pass = (lof_labels == 1)

            # --- DBSCAN ---
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_num_sample)
            dbscan_labels = dbscan.fit_predict(x_scaled)
            dbscan_valid = (dbscan_labels != -1)

            print(f"🐝 DBSCAN 有效 clients indices: {np.where(dbscan_valid)[0].tolist()}")
            print("------------------------------------------------------------")

        case _:
            raise Exception(f"Invalid cluster_method: {cluster_method}")
    
    # print(f'\n len x: {len(x)}')
    # print(f'\n labels: {labels}')
    # 計算每個群的大小
    unique_labels, counts = np.unique(labels, return_counts=True)
    # print(f'\n{[unique_labels, counts]}')

    # 排除噪音點的標籤 (-1)（僅適用於 DBSCAN）
    if 'DBSCAN' in cluster_method and -1 in unique_labels:
        noise_index = np.where(unique_labels == -1)
        unique_labels = np.delete(unique_labels, noise_index)
        counts = np.delete(counts, noise_index)
    
    if len(counts) == 0:
        print(f'DBSCAN No clusters found or all points are considered noise.\n')
        return

    # 找到最大的群
    max_label = unique_labels[np.argmax(counts)]
    # print(f'\n max_label: {max_label}')
    
    # 收集最大簇的成員及其權重
    max_cluster_members = {class_id: [] for class_id in classes}
    max_cluster_weights = {class_id: [] for class_id in classes}

    for idx, label in enumerate(labels):
        if label == max_label:
            class_id = y[idx]
            max_cluster_members[class_id].append(x[idx])
            max_cluster_weights[class_id].append(w[idx])

    # 定義加權平均函數
    def cluster_weighted_avg(members, weights):
        weights = torch.tensor(weights).view(-1, 1)
        weighted_sum = torch.sum(torch.stack(members) * weights, axis=0)
        sum_weights = torch.sum(weights)
        return weighted_sum / sum_weights

    # 聚合最大簇的成員並更新logit的weight和bias
    if agg_svc:
        with torch.no_grad():
            for class_id in classes:
                if max_cluster_members[class_id]:  # 確保該類別有成員
                    # new_wb = cluster_weighted_avg(max_cluster_members[class_id], max_cluster_weights[class_id])
                    new_wb = weighted_avg(max_cluster_members[class_id], max_cluster_weights[class_id],current_global_epoch)
                    global_model.logits.weight[class_id] = new_wb[:-1]
                    global_model.logits.bias[class_id] = new_wb[-1]

def FedGMMDBACG(global_model: torch.nn.Module, 
           client_models: list[torch.nn.Module], 
           client_weights: list[int], 
           global_optim: torch.optim, 
           logits_optim: torch.optim, 
           current_global_epoch: int, 
           num_global_epoch: int, 
           class_C: int | float, 
           base_agg: str, 
           agg_svc: bool, 
           spreadout: bool,
           cluster_method: str,
           num_clusters: int,
           dbscan_eps: int | float,
           dbscan_num_sample: int,
           # 新增高斯混合模型參數
           gmm_num_clusters: int = 2,  # 默認聚類數量為 2
           gmm_covariance_type: str = 'full',  # 默認設為 'full'
           gmm_tol: float = 1e-3,  # 默認容忍誤差為 1e-3
           gmm_max_iter: int = 100,  # 默認最大迭代次數為 100
           covariance_type: str = 'full',  # 默認設為 'full'
           tol: float = 1e-3,  # 默認容忍誤差為 1e-3
           max_iter: int = 100,  # 默認最大迭代次數為 100
           random_state: int = 0,  # 默認隨機種子
           client_epoch: int = 2,  # 每個客戶端的訓練輪數
           ) -> None:
    # 這裡是函數體，會根據具體需求執行相應的操作

    """
    Federated learning algorithm FedEFC.

    Arguments:
        global_model (torch.nn.Module): pytorch model (global model).
        client_models (list[torch.nn.Module]): pytorch models (client models).
        client_weights (list[int]): number of samples per client.
        clustering_num :預設K值
    """
    
    assert(base_agg == 'FedAvg' or base_agg == 'FedOpt')
    eval(base_agg)(global_model, client_models, client_weights, global_optim)
    
    # randomly select classes, whose embeddings will be updated using TurboSVM
    num_class = global_model.logits.weight.shape[0]
    if class_C <= 1.0: # proportion
        num_update_class = min(max(math.floor(class_C * num_class), 2), num_class)
    else: # class_C itself is num_update_class
        num_update_class = min(class_C, num_class)
    classes = np.random.choice(num_class, num_update_class, replace = False).tolist()
    classes.sort() # must be sorted when more than 2 classes, as svm coefs are sorted based on class id
    # print("number of SVC class:", num_update_class)
    # print(classes)
    # class embeddings, class labels, and client weights
    
    x, y, w = [], [], []
    for m, cw in zip(client_models, client_weights):
        wb = torch.cat((m.logits.weight[classes], m.logits.bias[classes].view(-1, 1)), axis = 1).detach()
        x.append(wb)
        y += classes
        w += [cw] * num_update_class
    x = torch.cat(x)
    assert(len(x) == len(y) == len(w)) # x, y, w should have same length
    # 確保 cluster_method 參數有正確傳遞
    print(f"DEBUG: cluster_method = {cluster_method}")

    
# 寫個高斯混合分布
# 新增惡意使用者的資料集，查看是否會影響到模型的訓練、預測、時間速度準確度等
    # 添加 'GaussianMixture' 的情況處理
    match cluster_method:
        case 'KMeans':
            # 對數據進行標準化處理
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            # 使用K-Means進行聚類
            kmeans = KMeans(n_clusters=num_clusters, max_iter=50, tol=1e-3)
            kmeans.fit(x_scaled)
            labels = kmeans.labels_

            # ✅ 統計每個群集的樣本數
            counts = np.bincount(labels)
            majority_cluster = np.argmax(counts)

            # ✅ 僅保留屬於最大群集的 client 為 valid
            valid_clients = (labels == majority_cluster)
            fallback_valid_clients = np.zeros(len(client_models), dtype=bool)  # 不使用 fallback

        case 'KMedoids':
            # 對數據進行標準化處理
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            # 使用K-Medoids進行聚類
            kmedoids = KMedoids(n_clusters=num_clusters, max_iter=50, method='pam', random_state=0)
            kmedoids.fit(x_scaled)
            labels = kmedoids.labels_

            # ✅ 選出最大群集作為有效 client
            counts = np.bincount(labels)
            majority_cluster = np.argmax(counts)
            valid_clients = (labels == majority_cluster)
            fallback_valid_clients = np.zeros(len(client_models), dtype=bool)

        case 'GaussianMixture':
            print("Running GaussianMixture...")
            # 使用 GaussianMixture 聚類
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            
            gmm = GaussianMixture(
                n_components=gmm_num_clusters,
                covariance_type=gmm_covariance_type,
                tol=gmm_tol,
                max_iter=gmm_max_iter,
                random_state=random_state
            )
            gmm.fit(x_scaled)

            means = gmm.means_
            covariances = gmm.covariances_
            probs = gmm.predict_proba(x_scaled)

            # 計算每個群集的集中度（協方差的行列式或對角線元素之和）
            concentration = np.array([np.sum(np.diagonal(cov)) for cov in covariances])
            threshold = 0.5
            max_prob = np.max(probs, axis=1)
            valid_samples = max_prob >= threshold

            # 篩選出高可信度樣本的標籤
            x_filtered = x_scaled[valid_samples]
            labels_filtered = gmm.predict(x_filtered)

            # 找出協方差小的可信群集
            good_clusters = concentration < np.percentile(concentration, 50)
            good_cluster_labels = np.where(good_clusters)[0]

            # 初始化為全 False
            valid_clients = np.zeros(len(client_models), dtype=bool)

            # 將原始樣本中屬於可信群集的且通過隸屬度閾值的 client 標為 True
            valid_indices = np.where(valid_samples)[0]
            for idx, cluster_label in zip(valid_indices, labels_filtered):
                if cluster_label in good_cluster_labels:
                    valid_clients[idx] = True

            fallback_valid_clients = np.zeros(len(client_models), dtype=bool)

        case 'HDBSCAN':
            # 對數據進行標準化處理
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)

            # 使用 HDBSCAN 進行聚類
            hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=dbscan_num_sample)
            labels = hdbscan_clusterer.fit_predict(x_scaled)

            # HDBSCAN 將噪聲點標記為 -1，其他為有效群集
            valid_clients = labels != -1
            fallback_valid_clients = np.zeros(len(client_models), dtype=bool)

        case 'None':
            # 無聚類方法，將全部數據視為一個簇
            labels = np.zeros(len(x), dtype=int)
            # 所有 client 都被視為有效
            valid_clients = np.ones(len(client_models), dtype=bool)
            fallback_valid_clients = np.zeros(len(client_models), dtype=bool)

        case 'GaussianMixtureDBSCAN':
            print("Running GMM-only anomaly detection...")

            # 高斯混合模型（GMM）初始化與篩選
            # step 1. 標準化數據： 這部分將每個客戶端模型的參數（parameters()）展平並拼接，然後對這些參數進行標準化處理，確保後續的異常偵測過程不會受異常範圍的影響。
            scaler = StandardScaler()
            x_raw = np.array([np.concatenate([p.data.cpu().numpy().flatten() for p in m.parameters()]) for m in client_models])
            x_scaled = scaler.fit_transform(x_raw)
            # step 2. PCA 降維：這部分將數據降維到 10 維，這樣可以減少計算量並提高 GMM 的性能。
            pca_dim = min(10, x_scaled.shape[0], x_scaled.shape[1])
            if x_scaled.shape[1] > pca_dim:
                pca = PCA(n_components=pca_dim)
                x_scaled = pca.fit_transform(x_scaled)

            valid_clients = np.full(len(x_scaled), True)  # 預設全部 valid
            labels = np.full(len(x_scaled), -1)

            # step 3. GMM 聚類：使用高斯混合模型（GMM）對客戶端模型進行聚類，通過計算每個模型屬於某一群集的機率，選擇出屬於主群集的模型。這是用來檢測是否存在異常數據點。
            gmm = GaussianMixture(
                n_components=min(gmm_num_clusters, len(x_scaled) // 2),
                covariance_type=gmm_covariance_type,
                tol=gmm_tol,
                max_iter=gmm_max_iter,
                random_state=random_state
            )
            gmm.fit(x_scaled)
            gmm_labels = gmm.predict(x_scaled)
            probs = gmm.predict_proba(x_scaled)
            max_prob = np.max(probs, axis=1)
            print("📊 GMM label 分佈:", np.bincount(gmm_labels))
            
            # 🧠 找出最大群集
            counts = np.bincount(gmm_labels)
            majority_cluster = np.argmax(counts)

            # ✅ 條件一：屬於最大群集
            in_majority = (gmm_labels == majority_cluster)

            # ✅ 條件二：機率在主群集的下分位門檻以上
            # ✅ 條件二：機率在主群集的中位數以上（更嚴格）
            prob_threshold = np.median(max_prob[in_majority])
            prob_pass = (max_prob >= prob_threshold)


            # ✅ 條件三：Mahalanobis距離 Z-score < 90%
            def mahalanobis_dist(x, mean, cov_inv):
                return distance.mahalanobis(x, mean, cov_inv)

            # step 4. Mahalanobis 距離計算：這部分計算每個客戶端模型到其所屬群集均值的 Mahalanobis 距離，並根據距離的分佈篩選出距離較近的模型。這是用來進一步檢測異常數據點。
            distances = np.array([
                mahalanobis_dist(x_scaled[i], gmm.means_[gmm_labels[i]], np.linalg.inv(gmm.covariances_[gmm_labels[i]]))
                for i in range(len(x_scaled))
            ])
            # z_scores = (distances - np.mean(distances)) / np.std(distances)
            # z_pass = np.abs(z_scores) < 1  # Z-score 小於 1（更合理界定）
            z_pass = distances < np.percentile(distances, 90)  # 保留前 90% 接近的客戶端

            # 來自 GMM 分群的分析結果。
            # 有效 client 要同時滿足：
            # 1️⃣ 屬於最大群集 (in_majority)
            # 2️⃣ 在該群集的分群機率高於 25% 分位 (prob_pass)
            # 3️⃣ Mahalanobis 距離的 Z-score < 1 (z_pass)
            # 🎯 GMM 最終 valid clients
            gmm_valid = in_majority & prob_pass & z_pass
            
            # 額外加入 weight norm 檢查（強化不一致異常偵測）
            weight_norms = np.linalg.norm(x_scaled, axis=1)
            norm_mean, norm_std = np.mean(weight_norms), np.std(weight_norms)
            norm_z = (weight_norms - norm_mean) / norm_std
            norm_pass = (np.abs(norm_z) < 2.5)

            # step 5.  Isolation Forest 和 LOF 進行異常檢測，額外異常檢測： 使用 Isolation Forest 和 Local Outlier Factor（LOF）進行進一步的異常偵測，這些方法能夠檢測到離群的數據點，並將其標記為異常。
            # 訓練 Isolation Forest 模型
            iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=random_state)
            iso_forest.fit(x_scaled)
            iso_labels = iso_forest.predict(x_scaled)
            iso_pass = (iso_labels == 1)  # 標記為正常的 client

            # 訓練 LOF (Local Outlier Factor)
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            lof_labels = lof.fit_predict(x_scaled)
            lof_pass = (lof_labels == 1)  # 標記為正常的 client 

            # step 6. DBSCAN 檢測：使用 DBSCAN 進行進一步的異常檢測，這部分將數據分為不同的簇，並將噪聲點標記為 -1。這是用來檢測是否存在異常數據點。
            # 進一步的 DBSCAN 檢測
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_num_sample)
            dbscan_labels = dbscan.fit_predict(x_scaled)

            # DBSCAN 噪聲標籤 -1 代表噪聲
            dbscan_valid = (dbscan_labels != -1)  # 只保留非噪聲點
            dbscan_valid_indices = np.where(dbscan_valid)[0].tolist()
            print(f"\n🐝 DBSCAN 有效 clients indices: {dbscan_valid_indices}")
            print("------------------------------------------------------------")

            # step 7. 最後的有效 client 條件：綜合多重條件篩選有效客戶端：這些條件包括 GMM 聚類、DBSCAN、Mahalanobis 距離、Isolation Forest、LOF 和其他異常檢測條件。
            # 最後的有效 client 條件：所有條件滿足才是有效
            # valid_clients = gmm_valid & dbscan_valid & norm_pass & iso_pass & lof_pass
            # 最後的有效 client 條件：所有條件滿足才是有效
            if dbscan_valid.sum() > len(client_models) / 2:
                valid_clients = gmm_valid & dbscan_valid & norm_pass & iso_pass & lof_pass
            elif gmm_valid.sum() > len(client_models) / 2:
                valid_clients = gmm_valid & norm_pass & iso_pass & lof_pass
            else:
                valid_clients = norm_pass & iso_pass & lof_pass

            # 後續處理
            filtered_client_models = [m for i, m in enumerate(client_models) if valid_clients[i]]
            filtered_client_weights = [w for i, w in enumerate(client_weights) if valid_clients[i]]

            if len(filtered_client_models) > 0:
                print("🔁 使用 valid clients 更新 global model")
                eval(base_agg)(global_model, filtered_client_models, filtered_client_weights, global_optim)
            # 若是沒有有效的可使用客戶端則進行 以下操作
            else:
                print("⚠️ 沒有 valid clients，改採用 GMM 最大群集進行 fallback 聚合")
                # step 1. fallback：選擇 GMM 最大群集作為 valid_clients
                counts = np.bincount(gmm_labels)
                majority_cluster = np.argmax(counts)
                fallback_valid_clients = (gmm_labels == majority_cluster)

                # step 2. 使用 fallback valid clients
                filtered_client_models = [m for i, m in enumerate(client_models) if fallback_valid_clients[i]]
                filtered_client_weights = [w for i, w in enumerate(client_weights) if fallback_valid_clients[i]]
                
                # step 3. 最終過濾與更新： 在過濾出有效的客戶端後，使用修剪平均方法對客戶端模型進行聚合，並更新全局模型。
                if len(filtered_client_models) > 0:
                    print("🔁 使用 fallback clients 更新 global model")
                    eval(base_agg)(global_model, filtered_client_models, filtered_client_weights, global_optim)
                    # 重建 x, y, w：為 agg_svc 做好準備
                    x, y, w = [], [], []
                    for m, cw in zip(filtered_client_models, filtered_client_weights):
                        wb = torch.cat((m.logits.weight[classes], m.logits.bias[classes].view(-1, 1)), axis=1).detach()
                        x.append(wb)
                        y += classes
                        w += [cw] * len(classes)
                    x = torch.cat(x)
                    labels = np.zeros(len(x), dtype=int)  # fallback 時可統一視為同一簇

                else:
                    print("⛔ fallback 聚合也失敗，跳過這一輪")

            # --- 計算出向量，使其不要有非典型梯度主導全局模型的情況---
            # --- 🧠 計算每個 client 的 gradient 向量與 global gradient 的 cosine similarity ---
            def model_to_vector(model, device=None):
                if device is None:
                    device = next(model.parameters()).device
                return torch.cat([p.data.view(-1).to(device) for p in model.parameters()])

            # global_vec = model_to_vector(global_model)

            # client_grads = []
            # cosine_similarities = []
            # # step 8.  計算並篩選梯度相似度：計算每個客戶端模型與全局模型的梯度向量之間的餘弦相似度，過濾掉與全局模型梯度差異過大的客戶端。
            # for m in client_models:
            #     client_vec = model_to_vector(m, device=global_vec.device)
            #     grad_vec = client_vec - global_vec  # 近似 gradient
            #     client_grads.append(grad_vec)
            #     cos_sim = 1 - cosine(global_vec.cpu().numpy(), grad_vec.cpu().numpy())
            #     cosine_similarities.append(cos_sim)

            # # --- 🔍 使用百分位方式過濾 cosine similarity 太低的 client ---
            # cos_threshold = np.percentile(cosine_similarities, 20)  # bottom 20% 視為異常
            # cos_valid = np.array(cosine_similarities) >= cos_threshold
            # print(f"🧭 Gradient Cosine Similarity threshold: {cos_threshold:.4f}")
            # print(f"✅ Cosine pass: {np.sum(cos_valid)} / {len(cos_valid)}")

            # # --- ✅ 最終有效 clients 結合所有條件 ---
            # valid_clients = valid_clients & cos_valid

            # --- 🧮 Trimmed Mean 聚合（防止極端值影響） ---
            # step 8. 使用 trimmed mean 聚合：這部分將所有有效的 client 模型進行 trimmed mean 聚合
            # 修剪平均聚合： 使用修剪平均（trimmed mean）方法進行聚合，這樣可以減少極端值對最終模型的影響。這是用來處理訓練過程中可能出現的噪聲或異常數據。
            if np.sum(valid_clients) > 0:
                print("📈 使用 trimmed mean 聚合")
                valid_models = [m for i, m in enumerate(client_models) if valid_clients[i]]
                # 將所有模型轉成向量形式
                model_vecs = torch.stack([model_to_vector(m) for m in valid_models])
                
                # 每個參數位置取 trimmed mean（剃除 top/bottom 10%）
                trimmed = trim_mean(model_vecs.numpy(), proportiontocut=0.1, axis=0)
                # 回傳到模型參數中（需要重建模型）
                offset = 0
                with torch.no_grad():
                    for p in global_model.parameters():
                        numel = p.data.numel()
                        p.data.copy_(torch.tensor(trimmed[offset:offset + numel]).view_as(p.data))
                        offset += numel
            else:
                print("⛔ 沒有 valid clients 通過 cosine 過濾")

            removed_indices = np.where(~valid_clients)[0].tolist()
            print(f"❌ 本輪被篩除的 clients indices: {removed_indices}")

            print(f"✅ Valid clients: {np.sum(valid_clients)} / {len(client_models)}")
            print(f"✅ GMM valid: {np.sum(gmm_valid)} / {len(gmm_valid)}")
            print(f"✅ Norm pass: {np.sum(norm_pass)} / {len(norm_pass)}")
            print(f"✅ ISO pass: {np.sum(iso_pass)} / {len(iso_pass)}")
            print(f"✅ LOF pass: {np.sum(lof_pass)} / {len(lof_pass)}")
            print(f"✅ DBSCAN pass: {np.sum(dbscan_valid)} / {len(dbscan_valid)}")

        case _:
            raise Exception(f"Invalid cluster_method: {cluster_method}")
        
    def init_logits_momentum_if_needed(global_model):
        if not hasattr(global_model, "logits_momentum"):
            global_model.logits_momentum = dict()
        if not hasattr(global_model, "logits_delta"):
            global_model.logits_delta = dict()
        if not hasattr(global_model, "global_momentum"):
            global_model.global_momentum = dict()

    init_logits_momentum_if_needed(global_model)

    # 使用歷史動量去更新client端的logits
    # 這段程式碼是在全局模型的分類器 (logits) 中，針對每一個類別 c，使用被篩選為有效的 client 模型進行聚合，更新每一類別的 logits 權重與偏差（bias）參數。具體過程如下：
    # 1.  判斷 agg_svc 是否啟用，代表是否要針對每個類別的 logits 進行聚合。
    if agg_svc:
        print("⚙️ 正在執行 agg_svc → 使用 valid client 聚合 class logits")
        # 2. 針對每個類別進行聚合
        # 從所有 client 模型中，取出該類別的 logits 權重與偏差（logits.weight[c] 和 logits.bias[c]），拼接為一個向量
        # 只挑選通過 valid_clients 的 client
        # 也取出對應的 client_weights 作為聚合時的權重
        with torch.no_grad():  # 禁用 gradient tracking
            for c in classes:
                # 嘗試從 valid_clients 中取得 logits
                logits_values = [
                    torch.cat((m.logits.weight[c], m.logits.bias[c].view(1)), dim=0).detach()
                    for i, m in enumerate(client_models) if valid_clients[i]
                ]
                logits_weights = [
                    client_weights[i] for i in range(len(client_models)) if valid_clients[i]
                ]

                # 若無 valid client，改用 GMM fallback clients
                if len(logits_values) == 0 :
                    print(f"⚠️ class {c} 沒有 valid clients，使用 fallback clients")
                    logits_values = [
                        torch.cat((m.logits.weight[c], m.logits.bias[c].view(1)), dim=0).detach()
                        for i, m in enumerate(client_models) if fallback_valid_clients[i]
                    ]
                    logits_weights = [
                        client_weights[i] for i in range(len(client_models)) if fallback_valid_clients[i]
                    ]

                # 若 fallback 仍無可用 clients，則完全跳過
                if len(logits_values) == 0:
                    print(f"⛔ class {c} 無法更新（無有效 clients）")
                    continue
                # 輸入：num_clients_total（N）

                # step 3. 使用 weighted_avg_with_momentum_ACG 函數做聚合
                # 正常執行更新
                updated = weighted_avg_with_momentum_ACG(
                    logits_values, logits_weights, c, global_model,
                    current_round=current_global_epoch,  # 👈 記得傳進來！
                    min_clients_threshold = max(7, int(len(client_models) * 1/2)),  # 👈 設定最小客戶端數量
                    local_epoch=client_epoch, 
                )
                global_model.logits.weight[c].copy_(updated[:-1])
                global_model.logits.bias[c].copy_(updated[-1])