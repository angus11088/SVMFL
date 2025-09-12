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
from sklearn.metrics import pairwise_distances_argmin_min
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

# =========================
# ✅ 1️⃣ 建立計數器
num_zscore_used = 0
num_cluster_used = 0

# =========================
#threshold=2.5 來源為 Z-score 應用廣泛，包含文獻 Detecting Anomalies using Z-Score、federated learning robust aggregation 也常見（例如 RobustFedAvg, Krum 等文獻背景均支持）。
def filter_outlier_clients(global_model, client_models, client_weights, method='zscore', threshold=2.5):
    deltas = []
    global_device = next(global_model.parameters()).device
    for model in client_models:
        squared_diff = 0.0
        #計算單一 client 在 local training 後，它的模型參數與當前 global 模型參數之間的「整體 L2 距離」
        #PyTorch 的 norm 是針對單個 tensor，所以若想計算多個 tensor 的「總 L2 norm」，需要先對每個 tensor 分別算 norm，然後平方加總，最後再開根號。
        for p_global, p_client in zip(global_model.parameters(), model.parameters()):
            p_client = p_client.to(global_device)
            #p_global是 global model 的「某一層」的參數 tensor，p_client 是 client local model 的「相同層」參數 tensor。
            squared_diff += torch.norm(p_global.data - p_client.data, p=2).item() ** 2
        deltas.append(squared_diff ** 0.5)
    #deltas 是一個 Python list，用來 收集每一個 client 訓練後的模型與 global 模型之間的距離。
    #因此最後 deltas 的長度 = client 數量；每一個元素 = 該 client 的「距離」。
    deltas = np.array(deltas)

    if method == 'zscore':
        mean = deltas.mean()
        #std 是 deltas 的 標準差 (Standard Deviation)，數學定義是所有值與平均值差異的平方和的平方根除以樣本數，反映資料分散程度。
        std = deltas.std()
        #np.abs(deltas - mean) 計算每個 client 距離與平均值的差距
        keep_mask = np.abs(deltas - mean) <= threshold * std if std != 0 else np.ones_like(deltas, dtype=bool)
    else:
        raise ValueError("未知過濾方式")

    all_filtered = not np.any(keep_mask)
    if all_filtered:
        print("⚠️ 所有客戶端皆被判定為異常，將跳過Z-score過濾，使用Kmeans聚類作為 fallback")
        return client_models, client_weights, True  # 保留 True 作為 fallback indicator


    filtered_models = [m for m, keep in zip(client_models, keep_mask) if keep]
    filtered_weights = [w for w, keep in zip(client_weights, keep_mask) if keep]
    removed_indices = [i for i, keep in enumerate(keep_mask) if not keep]
    print(f"🧹 Z-score剔除 {len(client_models) - len(filtered_models)} 個異常 client 更新 ")
    print(f"剔除的 client indices: {removed_indices}")
    return filtered_models, filtered_weights, False

def filter_outlier_clients_by_cluster(
    global_model,
    client_models,
    client_weights,
    num_clusters=2,
):
    import numpy as np
    import torch
    from sklearn.cluster import KMeans

    global_device = next(global_model.parameters()).device
    delta_vectors = []

    # 1️⃣ 計算每個 client 的 delta 向量
    for model in client_models:
        delta = []
        for p_global, p_client in zip(global_model.parameters(), model.parameters()):
            delta_tensor = (p_client.to(global_device).data - p_global.data).view(-1)
            delta.append(delta_tensor.cpu().numpy())
        delta_vectors.append(np.concatenate(delta))

    delta_vectors = np.stack(delta_vectors)

    # 2️⃣ KMeans 聚類
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(delta_vectors)

    # 3️⃣ 計算每群的平均梯度 norm
    cluster_norm_stats = []
    for cluster_id in range(num_clusters):
        cluster_points = delta_vectors[labels == cluster_id]
        if len(cluster_points) == 0:
            continue
        norms = np.linalg.norm(cluster_points, axis=1)
        mean_norm = np.mean(norms)
        cluster_norm_stats.append((cluster_id, mean_norm, len(cluster_points)))

    if not cluster_norm_stats:
        print("⚠️ 所有群皆為空群，跳過過濾，保留所有 client")
        return client_models, client_weights, True

    # 4️⃣ 選擇平均 norm 最小的群
    best_cluster, best_norm, best_size = min(cluster_norm_stats, key=lambda x: x[1])
    print(f"✅ 平均梯度 norm 最小群: {best_cluster}, Mean Norm={best_norm:.4f}, Size={best_size}")

    # 5️⃣ 過濾
    keep_mask = np.array([label == best_cluster for label in labels])
    filtered_models = [m for m, keep in zip(client_models, keep_mask) if keep]
    filtered_weights = [w for w, keep in zip(client_weights, keep_mask) if keep]

    if len(filtered_models) == 0:
        print("⚠️ 過濾後無 client，跳過過濾，保留所有 client")
        return client_models, client_weights, True

    if len(filtered_models) < num_clusters:
        print(f"⚠️ 過濾後 client 數({len(filtered_models)}) / num_clusters({num_clusters})")

    print(f"🧹 梯度 norm 聚類剔除 {len(client_models) - len(filtered_models)} 個異常 client")
    return filtered_models, filtered_weights, False

def robust_filter_outlier_clients(
    global_model,
    client_models,
    client_weights,
    method='zscore',
    threshold=2.5,
    num_clusters=2
):
    global num_zscore_used, num_cluster_used  # 宣告使用全域變數

    # 先執行 Z-score/IQR
    filtered_models, filtered_weights, all_filtered = filter_outlier_clients(
        global_model, client_models, client_weights,
        method=method, threshold=threshold
    )

    if all_filtered:
        # fallback 使用 cluster
        print("🔁 啟動 fallback：改用 cluster-based filtering")
        filtered_models, filtered_weights, _ = filter_outlier_clients_by_cluster(
            global_model,
            client_models,
            client_weights,
            num_clusters=num_clusters,
        )
        num_cluster_used += 1
        return filtered_models, filtered_weights, True

    else:
        num_zscore_used += 1
        return filtered_models, filtered_weights, False

# =========================
# ✅ 3️⃣ 提供統計查詢函式

def get_filter_stats():
    global num_zscore_used, num_cluster_used
    return {
        'filter_outlier_clients_used': num_zscore_used,
        'filter_outlier_clients_by_cluster_used': num_cluster_used
    }

def print_filter_stats():
    stats = get_filter_stats()
    print("📊=== Outlier Filtering 使用次數統計 ===")
    print(f"🔹 ZScore 使用次數: {stats['filter_outlier_clients_used']}")
    print(f"🔹 KMeans 使用次數: {stats['filter_outlier_clients_by_cluster_used']}")
    print("=======================================")


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
    wandb.log(wandb_log, step=-1)
    
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

        # ------------ 惡意客戶端標記 ------------
        num_clients = len(update_clients)
        num_malicious = 3  # 固定三個惡意客戶端
        malicious_indices = random.sample(range(num_clients), num_malicious)

        # 步驟 5：訓練客戶端
        # training
        if args.switch_FL == "FedEFC":
            # ✅ FedEFC 對應的原始 for 迴圈
            for client, client_model in zip(update_clients, client_models):
                previous_features = client.local_train(client_model, global_model, previous_features)

        # elif args.switch_FL == "FedGMMDBACG":
        #     # ✅ FedGMMDBACG 對應的新版本，會回傳 updated_model
        #     for i, (client, client_model) in enumerate(zip(update_clients, client_models)):
        #         updated_model, previous_features = client.local_train(client_model, global_model, previous_features)
        #         client_models[i] = updated_model  # ✅ overwrite 原本的 client_model
        elif args.switch_FL == "FedGMMDBACG" and args.malicious != 'None' and args.malicious != 'Weak' and args.malicious != 'Strong':
            num_clients = len(update_clients)
            num_malicious = 3  # 固定三個惡意客戶端
            malicious_indices = random.sample(range(num_clients), num_malicious)
            # 🔍 印出本輪的惡意客戶端索引或名稱
            print(f"🔴 本輪梯度惡意客戶端索引: {malicious_indices}")
            print("🔴 對應惡意客戶端名稱: ", [update_clients[i].client_name if hasattr(update_clients[i], 'client_name') else f"Client-{i}" for i in malicious_indices])
            for i, (client, client_model) in enumerate(zip(update_clients, client_models)):
                if i in malicious_indices and args.malicious != 'None' and args.malicious != 'Weak' and args.malicious != 'Strong':
                    malicious_flag = True
                    malicious_type = args.malicious  # 'GradientWeak' 或 'GradientStrong'
                else:
                    malicious_flag = False
                    malicious_type = 'None'

                updated_model, previous_features = client.local_train(
                    client_model,
                    global_model,
                    previous_features,
                    malicious=malicious_flag,
                    malicious_type=malicious_type
                )
                client_models[i] = updated_model  # 覆蓋更新後的模型
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
            # 若 client 數量 >= 16，則指定 50% 為惡意
            if num_clients >= 16:
                num_malicious = int(num_clients * 0.2)  # 20% 惡意
                # num_malicious = int(num_clients * 0.5)  # 50% 惡意
            else:
                # 否則使用最多 25% 的邏輯
                num_malicious = min(10, max(2, num_clients // 4))
            # num_malicious = min(3, max(2, num_clients // 4))

        malicious_indices = random.sample(range(num_clients), num_malicious)
        if args.malicious == 'None':
            # 如果 args.malicious 是 'None'，則不進行任何擾動
            print("✅ 本輪無惡意客戶端")
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
            #記取母數是誰，顯示出資料分布，證明惡意客戶端的存在，是否有被偵測到
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

                        # 數值穩定化（仍保留以避免 NaN 爆炸）
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
        elif args.switch_FL == 'FedEFC' or args.switch_FL == 'FedGMMDBACG':
            while continue_training:
                # 全局模型聚合
                # ✅ 在聚合前移除異常更新(梯度爆炸的class去除)
                client_models, client_weights, all_filtered = robust_filter_outlier_clients(
                    global_model, client_models, client_weights
                )

                # 步驟 8：模型聚合與更新
                # 根據不同的聯邦學習方法（FedEFC、FedGMMDBACG、TurboSVM 等），進行全局模型的聚合。
                eval(args.fed_agg)(
                    global_model, client_models, client_weights,  # 基本聯邦學習參數
                    global_optim,  # 用於 FedOpt（FedAdam 和 FedAMS）
                    logits_optim,  # 用於 FedAwS 和 TurboSVM
                    current_global_epoch,  args.class_C, args.base_agg, args.agg_svc, args.spreadout,  # 用於 TurboSVM
                    args.cluster_method, args.num_clusters, args.dbscan_eps, args.dbscan_num_sample, args.client_epoch,  args.global_epoch,# 用於 FedGMMDBACG
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
                print_filter_stats()
                # 結束訓練
                break
        else:
            # 預設執行 FedAvg 或其他 baseline 方法
            eval(args.fed_agg)(global_model, client_models, client_weights,
                            global_optim,
                            logits_optim,
                            current_global_epoch, args.global_epoch,
                            args.class_C, args.base_agg, args.agg_svc, args.spreadout)
        # 步驟 10：性能評估
        # performance metrics
        global_train_dataset = torch.utils.data.ConcatDataset([c.dataset for c in update_clients])
        global_train_loader  = torch.utils.data.DataLoader(global_train_dataset, batch_size = args.global_bs, shuffle = False, pin_memory = True)
        wandb_log = {}
        model_eval(global_model, global_train_loader, wandb_log, 'train/')
        model_eval(global_model, global_test_loader , wandb_log, 'test/' )
        wandb_log['epoch_train_times'] = epoch_train_times
        wandb.log(wandb_log, step=current_global_epoch)
       
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
        #    num_global_epoch: int, 
           class_C: int | float, 
           base_agg: str, 
           agg_svc: bool, 
           spreadout: bool,
           cluster_method: str,
           num_clusters: int,
           dbscan_eps: int | float,
           dbscan_num_sample: int,
           client_epoch: int,  # 每個客戶端的訓練輪數
           global_epoch: int,  # 全局訓練輪數
           # 新增高斯混合模型參數
           gmm_num_clusters: int = 2,  # 默認聚類數量為 2
           gmm_covariance_type: str = 'full',  # 默認設為 'full'
           gmm_tol: float = 1e-3,  # 默認容忍誤差為 1e-3
           gmm_max_iter: int = 100,  # 默認最大迭代次數為 100
           covariance_type: str = 'full',  # 默認設為 'full'
           tol: float = 1e-3,  # 默認容忍誤差為 1e-3
           max_iter: int = 100,  # 默認最大迭代次數為 100
           random_state: int = 0,  # 默認隨機種子
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

        case 'GaussianMixtureDBSCANISO':
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
                n_components=max(2, min(gmm_num_clusters, len(x_scaled)//2)),
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
            z_pass = distances < np.percentile(distances, 90)  # 保留前 90% 接近的客戶端

            # 來自 GMM 分群的分析結果。
            # 有效 client 要同時滿足：
            # 1️⃣ 屬於最大群集 (in_majority)
            # 2️⃣ 在該群集的分群機率高於 25% 分位 (prob_pass)
            # 3️⃣ Mahalanobis 距離的 Z-score < 1 (z_pass)
            # 🎯 GMM 最終 valid clients
            gmm_valid = in_majority & prob_pass & z_pass
            
            # step 5.  Isolation Forest 進行異常檢測，額外異常檢測： 使用 Isolation Forest 進行進一步的異常偵測，這些方法能夠檢測到離群的數據點，並將其標記為異常。
            # 訓練 Isolation Forest 模型
            iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=random_state)
            iso_forest.fit(x_scaled)
            iso_labels = iso_forest.predict(x_scaled)
            iso_pass = (iso_labels == 1)  # 標記為正常的 client

            # step 6. DBSCAN 檢測：使用 DBSCAN 進行進一步的異常檢測，這部分將數據分為不同的簇，並將噪聲點標記為 -1。這是用來檢測是否存在異常數據點。
            # 進一步的 DBSCAN 檢測
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_num_sample)
            dbscan_labels = dbscan.fit_predict(x_scaled)

            # DBSCAN 噪聲標籤 -1 代表噪聲
            dbscan_valid = (dbscan_labels != -1)  # 只保留非噪聲點

            # step 7. 最後的有效 client 條件：綜合多重條件篩選有效客戶端：這些條件包括 GMM 聚類、DBSCAN、Mahalanobis 距離、Isolation Forest、LOF 和其他異常檢測條件。
            # 最後的有效 client 條件：所有條件滿足才是有效
            if dbscan_valid.sum() >= len(client_models) / 2:
                valid_clients = gmm_valid & dbscan_valid & iso_pass
            else:
                valid_clients = gmm_valid & iso_pass

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

            remove_iso = np.where(~iso_pass)[0].tolist()
            remove_GMM = np.where(~gmm_valid)[0].tolist()
            remove_dbscan = np.where(~dbscan_valid)[0].tolist()
            # --- 統計與輸出 ---
            # 消溶實驗、證明GM、ISO、DBSCAN等方法的有效性
            removed_indices = np.where(~valid_clients)[0].tolist()
            print(f"❌ 本輪被篩除的 clients indices: {removed_indices}")
            print(f"✅ Valid clients: {np.sum(valid_clients)} / {len(client_models)}")
            print(f"✅ GMM valid: {np.sum(gmm_valid)} / {len(gmm_valid)}")
            print(f"❌ 本輪被篩除的 remove_GMM: {remove_GMM}")
            print(f"✅ ISO pass: {np.sum(iso_pass)} / {len(iso_pass)}")
            print(f"❌ 本輪被篩除的 ISO: {remove_iso}")
            print(f"✅ DBSCAN pass: {np.sum(dbscan_valid)} / {len(dbscan_valid)}")
            print(f"❌ 本輪被篩除的 DBSCAN: {remove_dbscan}")

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

            # --- DBSCAN ---
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_num_sample)
            dbscan_labels = dbscan.fit_predict(x_scaled)
            dbscan_valid = (dbscan_labels != -1)

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
        #    num_global_epoch: int, 
           class_C: int | float, 
           base_agg: str, 
           agg_svc: bool, 
           spreadout: bool,
           cluster_method: str,
           num_clusters: int,
           dbscan_eps: int | float,
           dbscan_num_sample: int,
           client_epoch: int,  # 每個客戶端的訓練輪數
           global_epoch: int,  # 全局訓練輪數
           # 新增高斯混合模型參數
           gmm_num_clusters: int = 2,  # 默認聚類數量為 2
           gmm_covariance_type: str = 'full',  # 默認設為 'full'
           gmm_tol: float = 1e-3,  # 默認容忍誤差為 1e-3
           gmm_max_iter: int = 100,  # 默認最大迭代次數為 100
           covariance_type: str = 'full',  # 默認設為 'full'
           tol: float = 1e-3,  # 默認容忍誤差為 1e-3
           max_iter: int = 100,  # 默認最大迭代次數為 100
           random_state: int = 0,  # 默認隨機種子
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

        #未有GMM
        case 'DBSCANISO':
            print("Running DBSCANISO...")
            scaler = StandardScaler()
            x_raw = np.array([np.concatenate([p.data.cpu().numpy().flatten() for p in m.parameters()]) for m in client_models])
            x_scaled = scaler.fit_transform(x_raw)
            # step 2. PCA 降維
            pca_dim = min(10, x_scaled.shape[0], x_scaled.shape[1])
            if x_scaled.shape[1] > pca_dim:
                pca = PCA(n_components=pca_dim)
                x_scaled = pca.fit_transform(x_scaled)

            valid_clients = np.full(len(x_scaled), True)  # 預設全部 valid
            labels = np.full(len(x_scaled), -1)

            # 訓練 Isolation Forest 模型
            iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=random_state)
            iso_forest.fit(x_scaled)
            iso_labels = iso_forest.predict(x_scaled)
            iso_pass = (iso_labels == 1)  # 標記為正常的 client

            # step 6. DBSCAN 檢測：使用 DBSCAN 進行進一步的異常檢測，這部分將數據分為不同的簇，並將噪聲點標記為 -1。這是用來檢測是否存在異常數據點。
            # 進一步的 DBSCAN 檢測
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_num_sample)
            dbscan_labels = dbscan.fit_predict(x_scaled)

            # DBSCAN 噪聲標籤 -1 代表噪聲
            dbscan_valid = (dbscan_labels != -1)  # 只保留非噪聲點

            # step 7. 最後的有效 client 條件：綜合多重條件篩選有效客戶端：這些條件包括 GMM 聚類、DBSCAN、Mahalanobis 距離、Isolation Forest、LOF 和其他異常檢測條件。
            # 最後的有效 client 條件：所有條件滿足才是有效
            if dbscan_valid.sum() > len(client_models) / 2:
                valid_clients = dbscan_valid & iso_pass
            else:
                valid_clients = iso_pass 

            # 後續處理
            filtered_client_models = [m for i, m in enumerate(client_models) if valid_clients[i]]
            filtered_client_weights = [w for i, w in enumerate(client_weights) if valid_clients[i]]

            if len(filtered_client_models) > 0:
                print("🔁 使用 valid clients 更新 global model")
                eval(base_agg)(global_model, filtered_client_models, filtered_client_weights, global_optim)

            removed_indices = np.where(~valid_clients)[0].tolist()
            remove_dbscan = np.where(~dbscan_valid)[0].tolist()
            remove_iso = np.where(~iso_pass)[0].tolist()
            print(f"❌ 本輪被篩除的 clients indices: {removed_indices}")

            print(f"✅ Valid clients: {np.sum(valid_clients)} / {len(client_models)}")
            print(f"✅ DBSCAN valid: {np.sum(dbscan_valid)} / {len(dbscan_valid)}")
            print(f"❌ 本輪被篩除的 DBSCAN: {remove_dbscan}")
            print(f"✅ ISO pass: {np.sum(iso_pass)} / {len(iso_pass)}")
            print(f"❌ 本輪被篩除的 ISO: {remove_iso}")

        #未有DBSCAN
        case 'GaussianMixtureISO':
            print("Running GaussianMixture...")
            scaler = StandardScaler()
            x_raw = np.array([np.concatenate([p.data.cpu().numpy().flatten() for p in m.parameters()]) for m in client_models])
            x_scaled = scaler.fit_transform(x_raw)
            # step 2. PCA 降維
            pca_dim = min(10, x_scaled.shape[0], x_scaled.shape[1])
            if x_scaled.shape[1] > pca_dim:
                pca = PCA(n_components=pca_dim)
                x_scaled = pca.fit_transform(x_scaled)

            valid_clients = np.full(len(x_scaled), True)  # 預設全部 valid
            labels = np.full(len(x_scaled), -1)

            # step 3. GMM 聚類
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

            z_pass = distances < np.percentile(distances, 90)  # 保留前 90% 接近的客戶端

            gmm_valid = in_majority & prob_pass & z_pass

            # step 5.  Isolation Forest 和 LOF 進行異常檢測，額外異常檢測： 使用 Isolation Forest 和 Local Outlier Factor（LOF）進行進一步的異常偵測，這些方法能夠檢測到離群的數據點，並將其標記為異常。
            # 訓練 Isolation Forest 模型
            iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=random_state)
            iso_forest.fit(x_scaled)
            iso_labels = iso_forest.predict(x_scaled)
            iso_pass = (iso_labels == 1)  # 標記為正常的 client

            valid_clients = gmm_valid & iso_pass 

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
                    x, y, w = [], [], []
                    for m, cw in zip(filtered_client_models, filtered_client_weights):
                        wb = torch.cat((m.logits.weight[classes], m.logits.bias[classes].view(-1, 1)), axis=1).detach()
                        x.append(wb)
                        y += classes
                        w += [cw] * len(classes)
                    x = torch.cat(x)
                    labels = np.zeros(len(x), dtype=int)

                else:
                    print("⛔ fallback 聚合也失敗，跳過這一輪")

            removed_indices = np.where(~valid_clients)[0].tolist()
            remove_GMM = np.where(~gmm_valid)[0].tolist()
            remove_iso = np.where(~iso_pass)[0].tolist()
            print(f"❌ 本輪被篩除的 clients indices: {removed_indices}")

            print(f"✅ Valid clients: {np.sum(valid_clients)} / {len(client_models)}")
            print(f"✅ GMM valid: {np.sum(gmm_valid)} / {len(gmm_valid)}")
            print(f"❌ 本輪被篩除的 remove_GMM: {remove_GMM}")
            print(f"✅ ISO pass: {np.sum(iso_pass)} / {len(iso_pass)}")
            print(f"❌ 本輪被篩除的 ISO: {remove_iso}")
        # 未有ISO
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
            z_pass = distances < np.percentile(distances, 90)  # 保留前 90% 接近的客戶端

            gmm_valid = in_majority & prob_pass & z_pass

            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_num_sample)
            dbscan_labels = dbscan.fit_predict(x_scaled)

            # DBSCAN 噪聲標籤 -1 代表噪聲
            dbscan_valid = (dbscan_labels != -1)  # 只保留非噪聲點
            dbscan_valid_indices = np.where(dbscan_valid)[0].tolist()

            # step 7. 最後的有效 client 條件：綜合多重條件篩選有效客戶端：這些條件包括 GMM 聚類、DBSCAN、Mahalanobis 距離、Isolation Forest、LOF 和其他異常檢測條件。
            # 最後的有效 client 條件：所有條件滿足才是有效
            if dbscan_valid.sum() > len(client_models) / 2:
                valid_clients = gmm_valid & dbscan_valid
            else:
                valid_clients = gmm_valid

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

            remove_GMM = np.where(~gmm_valid)[0].tolist()
            remove_dbscan = np.where(~dbscan_valid)[0].tolist()

            # --- 統計與輸出 ---
            # 消溶實驗、證明GM、DBSCAN等方法的有效性
            removed_indices = np.where(~valid_clients)[0].tolist()
            print(f"❌ 本輪被篩除的 clients indices: {removed_indices}")
            print(f"✅ Valid clients: {np.sum(valid_clients)} / {len(client_models)}")
            print(f"✅ GMM valid: {np.sum(gmm_valid)} / {len(gmm_valid)}")
            print(f"❌ 本輪被篩除的 remove_GMM: {remove_GMM}")
            print(f"✅ DBSCAN pass: {np.sum(dbscan_valid)} / {len(dbscan_valid)}")
            print(f"❌ 本輪被篩除的 DBSCAN: {remove_dbscan}")
        # All have
        case 'GaussianMixtureDBSCANISO':
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
                n_components=max(2, min(gmm_num_clusters, len(x_scaled)//2)),
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
            z_pass = distances < np.percentile(distances, 90)  # 保留前 90% 接近的客戶端

            # 來自 GMM 分群的分析結果。
            # 有效 client 要同時滿足：
            # 1️⃣ 屬於最大群集 (in_majority)
            # 2️⃣ 在該群集的分群機率高於 25% 分位 (prob_pass)
            # 3️⃣ Mahalanobis 距離的 Z-score < 1 (z_pass)
            # 🎯 GMM 最終 valid clients
            gmm_valid = in_majority & prob_pass & z_pass
            
            # step 5.  Isolation Forest 進行異常檢測，額外異常檢測： 使用 Isolation Forest 進行進一步的異常偵測，這些方法能夠檢測到離群的數據點，並將其標記為異常。
            # 訓練 Isolation Forest 模型
            iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=random_state)
            iso_forest.fit(x_scaled)
            iso_labels = iso_forest.predict(x_scaled)
            iso_pass = (iso_labels == 1)  # 標記為正常的 client

            # step 6. DBSCAN 檢測：使用 DBSCAN 進行進一步的異常檢測，這部分將數據分為不同的簇，並將噪聲點標記為 -1。這是用來檢測是否存在異常數據點。
            # 進一步的 DBSCAN 檢測
            dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_num_sample)
            dbscan_labels = dbscan.fit_predict(x_scaled)

            # DBSCAN 噪聲標籤 -1 代表噪聲
            dbscan_valid = (dbscan_labels != -1)  # 只保留非噪聲點

            # step 7. 最後的有效 client 條件：綜合多重條件篩選有效客戶端：這些條件包括 GMM 聚類、DBSCAN、Mahalanobis 距離、Isolation Forest、LOF 和其他異常檢測條件。
            # 最後的有效 client 條件：所有條件滿足才是有效
            if dbscan_valid.sum() >= len(client_models) / 2:
                valid_clients = gmm_valid & dbscan_valid & iso_pass
            else:
                valid_clients = gmm_valid & iso_pass

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

            remove_iso = np.where(~iso_pass)[0].tolist()
            remove_GMM = np.where(~gmm_valid)[0].tolist()
            remove_dbscan = np.where(~dbscan_valid)[0].tolist()
            # --- 統計與輸出 ---
            # 消溶實驗、證明GM、ISO、DBSCAN等方法的有效性
            removed_indices = np.where(~valid_clients)[0].tolist()
            print(f"❌ 本輪被篩除的 clients indices: {removed_indices}")
            print(f"✅ Valid clients: {np.sum(valid_clients)} / {len(client_models)}")
            print(f"✅ GMM valid: {np.sum(gmm_valid)} / {len(gmm_valid)}")
            print(f"❌ 本輪被篩除的 remove_GMM: {remove_GMM}")
            print(f"✅ ISO pass: {np.sum(iso_pass)} / {len(iso_pass)}")
            print(f"❌ 本輪被篩除的 ISO: {remove_iso}")
            print(f"✅ DBSCAN pass: {np.sum(dbscan_valid)} / {len(dbscan_valid)}")
            print(f"❌ 本輪被篩除的 DBSCAN: {remove_dbscan}")
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
                    current_round=current_global_epoch,  # 👈 輪次
                    min_clients_threshold = max(7, int(len(client_models) * 1/2)),  # 👈 設定最小客戶端數量
                    local_epoch=client_epoch, 
                    total_clients=len(client_models),
                    global_epoch = global_epoch, #總輪次
                )
                global_model.logits.weight[c].copy_(updated[:-1])
                global_model.logits.bias[c].copy_(updated[-1])