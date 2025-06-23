import torch
import copy
import numpy as np
import random
import argparse
import math
import wandb
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, Tuple, Union

from tabulate import tabulate
# self-defined functions
from client import get_clients
from clientACG import get_clients as get_clients_ACG
from models import CNN_femnist, CNN_celeba, LSTM_shakespeare, Resnet50_covid19
from data_preprocessing import get_data_dict_femnist, get_data_dict_celeba, get_data_dict_shakespeare, get_data_dict_covid19

def seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Arguments:
        seed (int): random seed.
    """

    print('\nrandom seed:', seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
def Args() -> argparse.Namespace:
    """
    Helper function for argument parsing.

    Returns:
        args (argparse.Namespace): parsed argument object.
    """

    parser = argparse.ArgumentParser()
    
    # path parameters
    parser.add_argument('--femnist_train_path', type = str, default = '../femnist/train/all_data_0_niid_0_keep_0_train_9.json', help = 'femnist train json path')
    parser.add_argument('--femnist_test_path' , type = str, default = '../femnist/test/all_data_0_niid_0_keep_0_test_9.json'  , help = 'femnist test json path')
    parser.add_argument('--celeba_train_path' , type = str, default = '../celeba/train/all_data_0_0_keep_5_train_9.json', help = 'celeba train json path')
    parser.add_argument('--celeba_test_path'  , type = str, default = '../celeba/test/all_data_0_0_keep_5_test_9.json'  , help = 'celeba test json path')
    parser.add_argument('--celeba_image_path' , type = str, default = '../celeba/img_align_celeba/', help = 'celeba image dir path')
    parser.add_argument('--shakespeare_train_path', type = str, default = '../shakespeare/train/all_data_0_0_keep_0_train_9.json', help = 'shakespeare train json path')
    parser.add_argument('--shakespeare_test_path' , type = str, default = '../shakespeare/test/all_data_0_0_keep_0_test_9.json'  , help = 'shakespeare test json path')
    parser.add_argument('--covid19_train_path', type = str, default = '../CC19/train/', help = 'covid19 train dir path')
    parser.add_argument('--covid19_test_path' , type = str, default = '../CC19/test/' , help = 'covid19 test dir path')

    # whether to use default settings for batch size and learning rates
    parser.add_argument('-d', '--default', type = bool, default = True, action = argparse.BooleanOptionalAction, help = 'whether to use default hyperparmeter settings (batch size and learning rates)')

    # general parameters for both non-FL and FL
    '''
    -p project name
    --name wandb名稱
    -seed 隨機種子seed
    --min_sample client的最小sample
    '''
    parser.add_argument('-p', '--project', type = str, default = 'femnist', help = 'project name, from femnist, celeba, shakespeare, covid19')
    parser.add_argument('--name', type = str, default = 'name', help = 'wandb run name')
    parser.add_argument('-seed', '--seed', type = int, default = 0, help = 'random seed')
    parser.add_argument('--min_sample', type = int, default = 64, help = 'minimal amount of samples per client')
    parser.add_argument('-g_bs', '--global_bs', type = int, default = 64, help = 'batch size for global data loader')
    parser.add_argument('-c_lr', '--client_lr', type = float, default = 1e-1, help = 'client learning rate')
    parser.add_argument('-global_epoch', '--global_epoch', type = int, default = 201, help = 'number of global aggregation rounds')
    parser.add_argument('--reuse_optim', type = bool, default = False, action = argparse.BooleanOptionalAction, help = 'whether to reuse client optimizer, should be T for non-fl and F for FL')
    parser.add_argument('-c_op', '--client_optim', default = torch.optim.SGD, help = 'client optimizer')
                    
    #C 8 16 32
    #E 1 2 4
    #TurboSVM=>SVM
    #DBSCAN =>eps 0.1 (0.5) 0.9,num_sample 1/3*C 1/2*C  SVM random

    # general parameters for FL
    parser.add_argument('-fl', '--switch_FL', type = str, default = 'FedAvg', help = 'FL algorithm, from FedAvg, FedAdam, FedAMS, FedProx, MOON, FedAwS, TurboSVM, FedEFC, FedGMMDBACG')
    parser.add_argument('-c_bs', '--client_bs', type = int, default = 64, help = 'batch size for client data loader')
    parser.add_argument('-C', '--client_C', type = int, default = 8, help = 'number of participating clients in each aggregation round')
    parser.add_argument('-E', '--client_epoch', type = int, default = 1, help = 'number of client local training epochs')
    parser.add_argument('-M', '--cluster_method', type=str, default='GaussianMixtureDBSCANISO', choices=['KMeans', 'KMedoids', 'DBSCAN', 'GaussianMixture', 'GaussianMixtureDBSCANISO', 'GMMDBSCAN', 'DBSCANISO', 'GaussianMixtureISO', 'GaussianMixtureDBSCAN', 'GaussianMixtureDBSCANISO-NoTrimmedMean'], help='Cluster algorithm, from KMeans, KMedoids, DBSCAN, GaussianMixture, DBSCANISO, GaussianMixtureISO, GMMDBSCAN, GaussianMixtureDBSCANISO, GaussianMixtureDBSCAN')
    parser.add_argument('-K', '--num_clusters', type = int, default = 4, help = 'number of k-means cluster')
    parser.add_argument('-eps', '--dbscan_eps', type = float, default = 0.5, help = 'eps of DBSCAN')
    parser.add_argument('-num_sample', '--dbscan_num_sample', type = int, default = 5, help = 'num_sample of DBSCAN')
    parser.add_argument('-min_cluster_size ', '--hdbscan_min_cluster_size', type = int, default = 5, help = 'min_cluster_size of HDBSCAN')
    #parser.add_argument('-C', '--client_C', type = int, default = 3221, help = 'number of participating clients in each aggregation round')
    #parser.add_argument('-E', '--client_epoch', type = int, default = 20, help = 'number of client local training epochs')
    
    # for FedOpt and FedAMS
    parser.add_argument('-g_lr', '--global_lr', type = float, default = 1e-3, help = 'global learning rate')
    parser.add_argument('-g_op', '--global_optim', default = torch.optim.Adam, help = 'global optimizer')
    
    # for TurboSVM
    parser.add_argument('--base_agg', type = str, default = 'FedAvg', help = 'basic aggregation method for non-logit layers for our method')
    parser.add_argument('--agg_svc', type = bool, default = True, action = argparse.BooleanOptionalAction, help = 'whether aggregating support vectors or all class embeddings for our method')
    parser.add_argument('--spreadout', type = bool, default = True, action = argparse.BooleanOptionalAction, help = 'whether conduing spread-out regularization for our method')
    parser.add_argument('--class_C', type = float, default = 1.0, help = 'proportion of classes being aggregated for our method')
    parser.add_argument('-l_lr', '--logits_lr', type = float, default = 1e-2, help = 'global learning rate for logit layer for our method')
    parser.add_argument('-l_op', '--logits_optim', default = torch.optim.Adam, help = 'global optimizer for logit layer for our method')
    
    # for FedEFC(GaussianMixture)、 FedGMMDBACG
    # **確保這些參數被解析**
    # 對應 GaussianMixture 特有的參數
    parser.add_argument('--gmm_num_clusters', type=int, default=5, help='number of clusters for GaussianMixture')
    parser.add_argument('--gmm_covariance_type', type=str, default='full', choices=['full', 'tied', 'diag', 'spherical'], help="Covariance type for GaussianMixture")
    parser.add_argument('--gmm_tol', type=float, default=1e-5, help="Tolerance for GaussianMixture convergence")
    parser.add_argument('--gmm_max_iter', type=int, default=300, help="Maximum iterations for GaussianMixture")
    parser.add_argument('--gmm_random_state', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--quantile_threshold', type=float, default=0.2, help='Quantile threshold for filtering low-prob GMM clients.')

    # for FedGMMDBACG

    parser.add_argument("--server_logits_mode", type=str, default="classwise", choices=["classwise", "global"])

    #惡意客戶模擬
    parser.add_argument('--malicious', type = str, default = 'None', choices=['None', 'Weak', 'Strong'], help = 'malicious for client')
    
    args = parser.parse_args()
    args.time = str(datetime.now())[5:-10]
    args.fed_agg = None
    args.MOON = False
    args.FedProx = False
    args.amsgrad = False

    return args

# def get_clients_and_model(args: argparse.Namespace) -> tuple[list[object], list[object], torch.nn.Module]:
#     """
#     Determine dataset and model based on project name.

#     Arguments:
#         args (argparse.Namespace): parsed argument object.

#     Returns:
#         train_clients (list[Client]): list of training clients.
#         test_clients (list[Client]): list of test/validation clients.
#         model (torch.nn.Module): pytorch model for the specific task.
#     """

#     match args.project:
#         case 'femnist':
#             train_data_dict = get_data_dict_femnist(args.femnist_train_path, args.min_sample)
#             test_data_dict  = get_data_dict_femnist(args.femnist_test_path , args.min_sample)
#             model = CNN_femnist(args)

#         case 'celeba':
#             train_data_dict = get_data_dict_celeba(args.celeba_train_path, args.celeba_image_path, args.min_sample)
#             test_data_dict  = get_data_dict_celeba(args.celeba_test_path , args.celeba_image_path, args.min_sample)
#             model = CNN_celeba(args)

#         case 'shakespeare':
#             train_data_dict = get_data_dict_shakespeare(args.shakespeare_train_path, args.min_sample)
#             test_data_dict  = get_data_dict_shakespeare(args.shakespeare_test_path , args.min_sample)
#             model = LSTM_shakespeare(args)

#         case 'covid19':
#             train_data_dict = get_data_dict_covid19(args.covid19_train_path, args.min_sample)
#             test_data_dict  = get_data_dict_covid19(args.covid19_test_path , args.min_sample)
#             model = Resnet50_covid19(args)

#         case _:
#             raise Exception("wrong project:", args.project)
        
#     # get client lists
#     train_clients = get_clients(args, train_data_dict) ; del train_data_dict
#     test_clients  = get_clients(args, test_data_dict ) ; del test_data_dict

#     # some print
#     print("utils.py ===> number of train clients:", len(train_clients))
#     print("utils.py ===> number of test  clients:", len(test_clients ))
#     print("utils.py ===> length of train dataset:", sum([c.num_sample for c in train_clients]))
#     print("utils.py ===> length of test  dataset:", sum([c.num_sample for c in test_clients ]))

#     return train_clients, test_clients, model

# 測試歷史動量的更新
def get_clients_and_model(args: argparse.Namespace) -> tuple[list[object], list[object], torch.nn.Module]:
    """
    Determine dataset and model based on project name.

    Arguments:
        args (argparse.Namespace): parsed argument object.

    Returns:
        train_clients (list[Client]): list of training clients.
        test_clients (list[Client]): list of test/validation clients.
        model (torch.nn.Module): pytorch model for the specific task.
    """

    match args.project:
        case 'femnist':
            train_data_dict = get_data_dict_femnist(args.femnist_train_path, args.min_sample)
            test_data_dict  = get_data_dict_femnist(args.femnist_test_path , args.min_sample)
            model = CNN_femnist(args)

        case 'celeba':
            train_data_dict = get_data_dict_celeba(args.celeba_train_path, args.celeba_image_path, args.min_sample)
            test_data_dict  = get_data_dict_celeba(args.celeba_test_path , args.celeba_image_path, args.min_sample)
            model = CNN_celeba(args)

        case 'shakespeare':
            train_data_dict = get_data_dict_shakespeare(args.shakespeare_train_path, args.min_sample)
            test_data_dict  = get_data_dict_shakespeare(args.shakespeare_test_path , args.min_sample)
            model = LSTM_shakespeare(args)

        case 'covid19':
            train_data_dict = get_data_dict_covid19(args.covid19_train_path, args.min_sample)
            test_data_dict  = get_data_dict_covid19(args.covid19_test_path , args.min_sample)
            model = Resnet50_covid19(args)

        case _:
            raise Exception("wrong project:", args.project)
        
    # get client lists
    if args.switch_FL == 'FedGMMDBACG':
        train_clients = get_clients_ACG(args, train_data_dict) ; del train_data_dict
        test_clients  = get_clients_ACG(args, test_data_dict ) ; del test_data_dict
        print("utils.py ===> FedGMMDBACG")
    else:
        train_clients = get_clients(args, train_data_dict) ; del train_data_dict
        test_clients  = get_clients(args, test_data_dict ) ; del test_data_dict
        print("使用原始客戶端，未有加速")

    # some print
    print("utils.py ===> number of train clients:", len(train_clients))
    print("utils.py ===> number of test  clients:", len(test_clients ))
    print("utils.py ===> length of train dataset:", sum([c.num_sample for c in train_clients]))
    print("utils.py ===> length of test  dataset:", sum([c.num_sample for c in test_clients ]))

    return train_clients, test_clients, model



def default_setting(args: argparse.Namespace) -> None:
    """
    Set batch sizes and learning rates according to the choice of dataset and federated learning algorithm.

    Arguments:
        args (argparse.Namespace): parsed argument object.
    """

    assert(args.default)

    match args.project:
        case 'femnist':
            args.min_sample = 64
            args.global_bs  = 64
            args.client_bs  = 64
            args.client_lr  = 1e-1
            args.global_lr  = 1e-3
            args.logits_lr  = 1e-2

        case 'celeba':
            args.min_sample = 8
            args.global_bs  = 8
            args.client_bs  = 8
            args.client_lr  = 1e-3
            args.global_lr  = 1e-3 # all global learning rates are bad here
            args.logits_lr  = 1e-2

        case 'shakespeare':
            args.min_sample = 64
            args.global_bs  = 64
            args.client_bs  = 64
            args.client_lr  = 1
            args.global_lr  = 1e-2
            args.logits_lr  = 1e-1

        case 'covid19':
            args.min_sample = 64
            args.global_bs  = 64
            args.client_bs  = 64
                
        case _:
            raise Exception("wrong project:", args.project)
        
def switch_FL(args: argparse.Namespace) -> None:
    """
    Set aggregation strategy according to the choice of federated learning algorithm.

    Arguments:
        args (argparse.Namespace): parsed argument object.
    """

    match args.switch_FL:

        case 'FedAvg':
            args.fed_agg = 'FedAvg'

        case 'FedAdam':
            args.fed_agg = 'FedOpt'

        case 'FedAMS':
            args.fed_agg = 'FedOpt'
            args.amsgrad = True
    
        case 'FedProx':
            args.fed_agg = 'FedAvg'
            args.FedProx = True

        case 'MOON':
            args.fed_agg = 'FedAvg'
            args.MOON = True

        case 'FedAwS':
            args.fed_agg = 'FedAwS'

        case 'TurboSVM':
            args.fed_agg = 'TurboSVM'
          
        case 'FedEFC':
            args.fed_agg = 'FedEFC'

        case 'FedGMMDBACG':
            args.fed_agg = 'FedGMMDBACG'

        case _:
            raise Exception("wrong switch_FL:", args.switch_FL)
    
# def weighted_avg_params(params: list[dict[str, torch.Tensor]], weights: list[int] = None) -> dict[str, torch.Tensor]:
#     """
#     Compute weighted average of client models.

#     Argument:
#         params (list[dict[str, torch.Tensor]]): client model parameters. Each element in this list is the state_dict of a client model.
#         weights (list[int]): weight per client. Each element in this list is the number of samples of a client.

#     Returns:
#         params_avg (dict[str], torch.Tensor): averaged global model parameters (state_dict), which can be loaded using global_model.load_state_dict.
#     """

#     if weights == None:
#         weights = [1.0] * len(params)
        
#     params_avg = copy.deepcopy(params[0])
#     for key in params_avg.keys():
#         params_avg[key] *= weights[0]
#         for i in range(1, len(params)):
#             params_avg[key] += params[i][key] * weights[i]
#         params_avg[key] = torch.div(params_avg[key], sum(weights))
#     return params_avg

def weighted_avg_params(params: list[dict[str, torch.Tensor]], weights: list[int] = None) -> dict[str, torch.Tensor]:
    """
    Compute weighted average of client models.
    """
    if not params or len(params) == 0:
        print("[WARNING] weighted_avg_params: 收到空的 client 參數，回傳 None")
        return None
    
    if weights is None:
        weights = [1.0] * len(params)

    if len(weights) != len(params):
        raise ValueError("[ERROR] weighted_avg_params: weights 和 params 長度不一致！")

    params_avg = copy.deepcopy(params[0])
    for key in params_avg.keys():
        params_avg[key] *= weights[0]
        for i in range(1, len(params)):
            params_avg[key] += params[i][key] * weights[i]
        params_avg[key] = torch.div(params_avg[key], sum(weights))
    return params_avg

def weighted_avg(values: any, weights: any, current_global_epoch: int) -> any:
    """
    Calculate weighted average of a vector of values.

    Arguments:
        values (any): values. Can be list, torch.Tensor, numpy.ndarray, etc.
        weights (any): weights. Can be list, torch.Tensor, numpy.ndarray, etc.

    Returns:
        any: weighted average value.
    """
    # file = open(f'./weight/weight-{current_global_epoch}.log', 'a')
    # file.write("Values:\n")
    # for value in values:
    #     file.write(f"{np.array(value)}\n")

    # file.write("Weights:\n")
    # for weight in weights:
    #     file.write(f"{np.array(weight)}\n")
    # file.close()

    sum_values = 0
    for v, w in zip(values, weights):
        sum_values += v * w
    return sum_values / sum(weights)

# 歷史動量新增之後的方法-更改client model使其更加靠近正解
def weighted_avg_with_momentum_forType(
    values: Union[list[torch.Tensor], torch.Tensor],
    weights: list[float],
    class_id: int,
    global_model: torch.nn.Module,
    momentum_beta: float = 0.9
) -> torch.Tensor:
    """
    使用 FedACG server-side momentum 計算 logits 含 weight+bias 的加權平均。

    Arguments:
        values: list of Tensor 或單一 Tensor，皆為 logits (含 weight + bias)，shape = [n, dim] or [dim]
        weights: 對應的權重，長度為 n
        class_id: 當前類別 ID
        global_model: 全域模型，包含 logits.weight / bias
        momentum_beta: 動量參數 β

    Returns:
        updated: Tensor of shape [dim]，聚合後 logits（含 weight + bias）
    """
    device = global_model.logits.weight.device

    # 標準化 values → [n, dim]
    if isinstance(values, list):
        values = torch.stack(values).to(device)  # [n, dim]
    else:
        values = values.to(device)
        if values.dim() == 1:  # 單一向量 → 增一維
            values = values.unsqueeze(0)

    # 標準化 weights → [n, 1]
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    weights = weights / weights.sum()
    weights = weights.view(-1, 1)  # [n, 1]

    # 加權平均 logits
    avg = torch.sum(values * weights, dim=0)  # [dim]

    # 擷取舊的 logits（含 bias）為 baseline
    old_weight = global_model.logits.weight[class_id]  # [dim-1]
    old_bias = global_model.logits.bias[class_id].unsqueeze(0)  # [1]
    old_wb = torch.cat([old_weight, old_bias])  # [dim]

    # 初始化 delta / momentum
    if class_id not in global_model.logits_momentum:
        global_model.logits_momentum[class_id] = torch.zeros_like(avg)
    if class_id not in global_model.logits_delta:
        global_model.logits_delta[class_id] = torch.zeros_like(avg)

    # 更新 momentum 機制
    delta = avg - old_wb
    global_model.logits_delta[class_id] = delta
    global_model.logits_momentum[class_id] = (
        momentum_beta * global_model.logits_momentum[class_id] + delta
    )

    # 回傳更新值
    updated = old_wb + global_model.logits_momentum[class_id]  # [dim]
    return updated
#----------------------------------------
    # avg.norm	本輪平均向量的 norm
    # adjusted_max	根據歷史 norm 與 round 動態調整後的最大 norm
    # Δnorm	avg 與 lookahead 差距大小
    # mom_norm	momentum 向量 norm
    # α	最終更新比例
    
    # avg	客戶端傳回值加權平均後的結果（含權重加總、值 clip 處理），作為主候選的模型更新依據。
    # old_wb	舊的 logits 權重與 bias 拼接後的張量，作為 momentum、微縮、fallback 的 baseline。
    # delta	avg - old_wb，即本輪與上一輪參數的差值，用來計算 momentum 動量的基礎方向。
    # client_scale_ratio	描述目前客戶端數在合理區間內的比例（最多 0.778），作為調整參數比例的主控變數。
    # scale_factor	對 delta 的縮放倍率，根據 client_scale_ratio 與 local_epoch_scale 調整，用來控制更新強度。
    # base_beta	Momentum 基礎 beta 值（衰減比率），根據 client_scale_ratio 調整，決定歷史與當前權重的比。
    # delta_clip	對 delta 的 norm 限制上限，避免一次變化過大，依據 client 數與 local_epoch 調整。
    # max_mom_norm	限制 momentum 的最大 norm，若超出則進行 clip，確保動量不會爆炸。
    # momentum_beta	實際動量係數 beta，會根據當前 round 接近 exit threshold 而線性衰減（early-stage 增強）。
    # local_epoch_scale	根據本地訓練 epoch 數反映的穩定性係數（epoch 多 -> 調小更新強度），用於調節 scale 與 clip。
    # delta_clip (重複項)	見上面，經過 local_epoch_scale 微調後再次使用，反映最終 delta norm 限制。
    # scale_factor (重複項)	見上面，同樣會經過 local_epoch_scale 調整，用於對 delta 的最終縮放。
#----------------------------------------
already_logged_rounds = set()
# 目前論文實用版
# 結合動量機制與多種安全防護條件來穩定地執行多客戶端的參數加權聚合
def weighted_avg_with_momentum_ACG(
    values: Union[list[torch.Tensor], torch.Tensor],
    weights: list[float],
    class_id: int,
    global_model: torch.nn.Module,
    momentum_beta: float = 0.9,
    current_round: int = 0,
    min_clients_threshold: int = 0,
    momentum_exit_round: int = 125,
    local_epoch: int = 0,
    global_epoch: int = 0,
    total_clients: int = 0
) -> torch.Tensor:
    # step 1. 初步處理與加權平均計算：接收來自客戶端的權重與參數，並進行 clip 與 norm 檢查。
    # 這段程式碼的目的是進行加權平均操作，將來自不同客戶端的 values 根據其相應的權重 weights 加權，並計算出加權平均值 avg
    # 設定設備和客戶端數量
    device = global_model.logits.weight.device
    num_clients = len(weights)

    #  檢查並處理 values
    # 如果 values 是一個列表（list），則將其轉換為 PyTorch 張量並堆疊（torch.stack），同時確保其在正確的設備上。
    # 如果 values 是一維張量（dim() == 1），則會通過 unsqueeze(0) 在第一維上增加一個維度，使其變為二維張量（這樣便於後續的矩陣運算）。
    if isinstance(values, list):
        values = torch.stack(values).to(device)
    else:
        values = values.to(device)
        if values.dim() == 1:
            values = values.unsqueeze(0)

    # 處理 weights 並計算加權平均
    # 將 weights 轉換為 PyTorch 張量，確保其數據類型為 float32 並位於正確的設備上。
    # 將 weights 按總和進行歸一化，使其總和為 1（這是常見的加權操作）。這樣每個客戶端的權重會按比例進行調整。
    # 使用 view(-1, 1) 將 weights 轉換為列向量，這樣便於後續的矩陣運算。
    # 最後，計算加權平均值 avg，通過對 values 和 weights 的逐元素乘法後求和，dim=0 表示對第一個維度進行求和，從而得到加權平均值。
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    weights = weights / weights.sum()
    weights = weights.view(-1, 1)
    avg = torch.sum(values * weights, dim=0)

    # 這段程式碼的目的是確保在模型更新過程中，對於不穩定或過大的參數變動進行控制。具體來說，會根據範數（norm）的值來進行縮放、回退操作，避免過度或無效的更新。
    # 計算目標範數並準備權重和偏置
    # 定一個目標範數 target_norm，用於控制更新的大小，避免過大的更新。
    # 構建 weight_key 和 bias_key，用於提取指定 class_id 的權重和偏置。
    # 根據是否存在 global_momentum，選擇使用提前計算的動量（lookahead_weight 和 lookahead_bias）還是當前的權重和偏置。
    weight_key = f"logits.weight.{class_id}"
    bias_key = f"logits.bias.{class_id}"

    if hasattr(global_model, "global_momentum"):
        lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
        lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
        old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
    else:
        old_wb = torch.cat([global_model.logits.weight[class_id], global_model.logits.bias[class_id].unsqueeze(0)])
    
    # 如果 old_wb 本身就包含了 NaN 或 Inf，則會打印錯誤信息，表示該權重包含無效數據，這樣的情況會導致訓練無法正常進行。
    if torch.isnan(old_wb).any() or torch.isinf(old_wb).any():
        print(f"[ERROR] old_wb 本身已經含有 NaN 或 Inf！")

    # step 5. 動量條件與 early stage momentum 判斷：在特定條件下啟用 momentum（動量）更新。
    # 判斷是否進入 early_stage_momentum（早期階段的動量更新）
    # early_stage_momentum 的邏輯： early_stage_momentum 用來控制在客戶端數量較少、訓練輪次較少的情況下，是否啟用動量更新。這樣的設計有助於在早期階段穩定訓練過程，避免過早引入過多的動量，從而影響模型的更新。
    early_stage_momentum = (
        values.size(0) < min_clients_threshold and
        current_round < momentum_exit_round and
        local_epoch >= 4
    )
    # avg_norm 計算的是 avg 向量的範數（即它的模長），通常用來衡量模型權重的大小。這是後續調整更新的關鍵數據。
    avg_norm = avg.norm().item()
    print(f"[WARNING] round={current_round} class={class_id} avg_norm = {avg_norm:.4f}")
    # 如果客戶端數量少於 min_clients_threshold，並且不處於早期階段的動量更新（early_stage_momentum 為 False），則會跳過動量更新。
    if values.size(0) < min_clients_threshold or early_stage_momentum:
        print(f"[INFO] round={current_round} class={class_id} | client 太少或 early stage（{values.size(0)}），跳過 momentum")
        return avg.detach().clone()
    
    # 如果 avg_norm（即平均範數）小於 0，或者是 NaN 或 inf（即無效數值），則會打印警告並回退到先前的權重 old_wb。這樣做是為了防止使用無效的更新來更新模型。
    if avg_norm <= 0 or math.isnan(avg_norm) or math.isinf(avg_norm):
        print(f"[WARNING] round={current_round} class={class_id} | avg 無效，fallback to old_wb")
        return old_wb.detach().clone()

    # 如果 avg_norm 是有效的，並且進入了動量更新階段，則計算 delta，即當前平均權重 avg 和先前權重 old_wb 之間的差異。這個 delta 代表了權重的變化量，通常在動量更新中會被用來調整模型參數。
    delta = avg - old_wb
    
    #客戶參與度
    client_scale_ratio = num_clients / total_clients

    # 梯度裁剪上限（Round數越多、參與越多，允許越大更新）
    delta_clip = compute_delta_clip_dynamic(num_clients, total_clients, current_round)
    
    # 對 delta 進行裁剪：
    # 這裡計算了 delta（即梯度或更新量）的 L2 norm，並對 delta 進行裁剪，防止過大的更新。
    # 若 delta 的 norm 超過設定的 delta_clip，則將其縮放，使其 norm 恢復到 delta_clip 的範圍內，從而防止梯度爆炸。
    delta_norm_val = delta.norm()
    if delta_norm_val > delta_clip:
        print(f"[CLIP] round={current_round} class={class_id} | delta norm ({delta_norm_val:.4f}) > {delta_clip}, 進行 clip")
        delta = delta * (delta_clip / delta_norm_val)

    # 執行的是一種「momentum-based 更新機制」，目的是穩定 logits 層（分類層）參數的更新，並包含 clipping、防呆與 adaptive learning。
    # 檢查 global_model 是否有 logits_momentum 屬性，如果沒有就新增。
    # 每個 class（由 class_id 決定）都會各自維護一個 momentum 向量，初始為全零張量。
    # delta 是這一輪對 logits 層參數的變動量（可能是梯度、或模型差值）。
    if not hasattr(global_model, "logits_momentum"):
        global_model.logits_momentum = {}
    if class_id not in global_model.logits_momentum:
        global_model.logits_momentum[class_id] = torch.zeros_like(delta)
    
    # 計算新的 momentum
    # 使用指數移動平均法更新 momentum：
    # momentum_beta 控制慣性大小（類似 Adam 優化器的 β 參數）。這個式子會讓過去的方向影響現在的更新，使更新更平滑、更穩定。
    # prev_momentum 是之前累積的權重變化趨勢
    # delta 是目前新觀測到的權重變化
    # momentum_beta 控制「我們要記得多少過去的資訊」，而 (1 - momentum_beta) 則控制「我們要多快適應新的資訊」。
    # 設定代表 90% 來自過去動量的影響，10% 來自新的變化
    prev_momentum = global_model.logits_momentum[class_id]
    new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta

    mom_norm = new_momentum.norm()
    # 儲存更新後的 momentum
    global_model.logits_momentum[class_id] = new_momentum

    avg_norm = avg.norm().item()
    # 存儲歷史的 avg_norm
    # 這段程式碼會將當前的 avg_norm 存儲到 global_model.history_norms 列表中，以便跟踪過去 15 次更新中的範數變化。
    # 如果歷史範數列表的長度超過 15，會刪除最舊的範數值，保持列表長度為 15。
    if not hasattr(global_model, "history_norms"):
        global_model.history_norms = []
    global_model.history_norms.append(avg_norm)
    if len(global_model.history_norms) > 15:
        global_model.history_norms.pop(0)

    # 基於歷史範圍調整 max_avg_norm
    min_avg_norm = min(global_model.history_norms)
    max_avg_norm = max(global_model.history_norms)

    # 根據訓練過程進行自適應調整
    dynamic_range = max_avg_norm - min_avg_norm
    adjusted_max_avg_norm = min_avg_norm + dynamic_range

    # 剪裁 avg.norm
    if avg_norm > adjusted_max_avg_norm:
        print(f"[CLIP] round={current_round} | avg.norm={avg_norm:.4f} > {adjusted_max_avg_norm:.4f}, applying clip.")
        avg = avg * (adjusted_max_avg_norm / avg.norm())
    # 計算 adjusted_max_avg_norm，作為動態調整的上限參考
    # 自適應調整 blending factor alpha
    # alpha 是 logits 更新中，momentum 權重的參數：
    # avg_norm_ratio 控制更新幅度（norm 太大就降低比例）。
    avg_norm_ratio = min(1.0, adjusted_max_avg_norm / avg_norm)
    # 線性成長的 alpha_base，根據 round 數提升
    # 到了訓練中期以後（約一半輪次），模型參數大致穩定，此時就可以更信任過去的 momentum 更新，早期保守更新，後期穩定收斂
    # 客戶參與度縮放因子
    client_scale_ratio = num_clients / total_clients  # 一定是 < 1 的比例

    # 這裡的 alpha_base 是一個基礎值，隨著訓練進行而線性增長，並且不會超過 1.0。
    # 這樣可以在早期階段保守更新，後期則更信任過去的 momentum 更新。
    alpha_base = min(current_round / (global_epoch / 2), 1.0)
    # alpha 越大代表「越相信 momentum」。
    # 最終 alpha
    alpha = alpha_base * client_scale_ratio * avg_norm_ratio

    wandb.define_metric("current_round")

    if current_round not in already_logged_rounds:
        wandb.log({
            "current_round": current_round,
            "client_scale_ratio": float(client_scale_ratio),
            "delta_clip": float(delta_clip),
            "delta_norm": float(delta.norm().item()),
            "avg_norm": float(avg_norm),
            "adjusted_max_avg_norm": float(adjusted_max_avg_norm),
            "mom_norm": float(mom_norm),
            "alpha_base": float(alpha_base),
            "alpha": float(alpha),
        }, step=current_round)

        already_logged_rounds.add(current_round)
    # updated 是最終版本，送回去作為新的 logits 層參數。
    # 最終 logits 層參數更新：混合使用兩個資訊：1.avg：整體平均參數（較穩定）。 2.old_wb + new_momentum：加權的慣性更新方向（包含當前變化趨勢）。
    updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

    # 這段將 momentum 的權重部分和 bias 部分分別儲存起來，方便後續模型更新使用。
    if not hasattr(global_model, "global_momentum"):
        global_model.global_momentum = {}
    global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
    global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()
    
    # step 13. 回傳最終 updated 參數
    return updated

def compute_delta_clip_dynamic(num_clients, total_clients, current_round):

    client_ratio = num_clients / total_clients    #客戶參與率
    scaled_round = current_round / total_clients  # 將 current_round 與 total_clients 比例化，避免過大數值
    # 讓函數在 x → 0 附近不會趨近於負無限，而是有一個穩定的起點
    round_factor = math.log(1 + math.exp(scaled_round))
    delta_clip = client_ratio * round_factor

    return delta_clip

