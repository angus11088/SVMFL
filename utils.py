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
    parser.add_argument('-M', '--cluster_method', type=str, default='KMeans', choices=['KMeans', 'KMedoids', 'DBSCAN', 'GaussianMixture', 'GaussianMixtureDBSCAN', 'GMMDBSCAN'], help='Cluster algorithm, from KMeans, KMedoids, DBSCAN, GaussianMixture')
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
    parser.add_argument('--gmm_num_clusters', type=int, default=4, help='number of clusters for GaussianMixture')
    parser.add_argument('--gmm_covariance_type', type=str, default='full', choices=['full', 'tied', 'diag', 'spherical'], help="Covariance type for GaussianMixture")
    parser.add_argument('--gmm_tol', type=float, default=1e-3, help="Tolerance for GaussianMixture convergence")
    parser.add_argument('--gmm_max_iter', type=int, default=300, help="Maximum iterations for GaussianMixture")
    parser.add_argument('--gmm_random_state', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--quantile_threshold', type=float, default=0.2, help='Quantile threshold for filtering low-prob GMM clients.')

    # for FedGMMDBACG
    parser.add_argument('--momentum', type=float, default=0.0, help="Momentum factor for client updates (default: 0.0)")
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
        print("使用原始客戶端，未有AVG加速")

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

# 歷史動量的更新初始化-使用client model的logits進行初始化 - celeba成效好的
# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device

#     # ===== 將輸入標準化為 [n, dim] =====
#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)

#     avg = torch.sum(values * weights, dim=0)
#     # ⭐ 不對 avg 做 clamp（保留更自然的訊號變化）
    
#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     # ===== 取出 lookahead baseline（從 global momentum 中提取） =====
#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(
#             weight_key, global_model.logits.weight[class_id]
#         ).clone().detach().to(device)

#         lookahead_bias = global_model.global_momentum.get(
#             bias_key, global_model.logits.bias[class_id]
#         ).clone().detach().to(device)

#         old_wb = torch.cat([lookahead_weight, lookahead_bias.unsqueeze(0)])
#     else:
#         old_wb = torch.cat([
#             global_model.logits.weight[class_id],
#             global_model.logits.bias[class_id].unsqueeze(0)
#         ])

#     delta = avg - old_wb
#     # ✅ 初期縮放避免爆衝，後期穩定強化
#     if current_round < 100:
#         scale_factor = 0.3
#         delta_clip = 2.0
#         momentum_beta = 0.9
#         max_mom_norm = 20.0
#     elif current_round < 200:
#         scale_factor = 0.5
#         delta_clip = 5.0
#         momentum_beta = 0.85
#         max_mom_norm = 30.0
#     else:
#         scale_factor = 0.7
#         delta_clip = 6.0
#         momentum_beta = 0.8
#         max_mom_norm = 30.0

#     delta = torch.clamp(delta, min=-delta_clip, max=delta_clip)
#     delta = delta * scale_factor

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     # ===== 更新 momentum =====
#     global_model.logits_momentum[class_id] = (
#         momentum_beta * global_model.logits_momentum[class_id] +
#         (1 - momentum_beta) * delta
#     )

#     # ===== Clip momentum norm =====
#     norm = global_model.logits_momentum[class_id].norm()
#     if norm > max_mom_norm:
#         global_model.logits_momentum[class_id] *= (max_mom_norm / norm)

#     # ===== α 線性成長控制 momentum 更新強度 =====
#     alpha = min(0.1 + 0.0032 * current_round, 0.67)

#     updated = (1 - alpha) * avg + alpha * (old_wb + global_model.logits_momentum[class_id])

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}

#     global_model.global_momentum[weight_key] = global_model.logits_momentum[class_id][:-1].detach().clone()
#     global_model.global_momentum[bias_key] = global_model.logits_momentum[class_id][-1].detach().clone()

#     print(f"[DEBUG] round={current_round:3d} | Δnorm={delta.norm():.4f} | mom_norm={norm:.4f} | α={alpha:.2f}")
#     return updated


# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device

#     # ===== 將輸入標準化為 [n, dim] =====
#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)

#     avg = torch.sum(values * weights, dim=0)
#     # ⭐ 不對 avg 做 clamp（保留更自然的訊號變化）
    
#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     # ===== 取出 lookahead baseline（從 global momentum 中提取） =====
#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(
#             weight_key, global_model.logits.weight[class_id]
#         ).clone().detach().to(device)

#         lookahead_bias = global_model.global_momentum.get(
#             bias_key, global_model.logits.bias[class_id]
#         ).clone().detach().to(device)

#         old_wb = torch.cat([lookahead_weight, lookahead_bias.unsqueeze(0)])
#     else:
#         old_wb = torch.cat([
#             global_model.logits.weight[class_id],
#             global_model.logits.bias[class_id].unsqueeze(0)
#         ])

#     delta = avg - old_wb
#     # ✅ 初期縮放避免爆衝，後期穩定強化
#     if current_round < 100:
#         scale_factor = 0.3
#         delta_clip = 2.0
#         momentum_beta = 0.9
#         max_mom_norm = 20.0
#     elif current_round < 200:
#         scale_factor = 0.5
#         delta_clip = 5.0
#         momentum_beta = 0.85
#         max_mom_norm = 30.0
#     else:
#         scale_factor = 0.7
#         delta_clip = 6.0
#         momentum_beta = 0.8
#         max_mom_norm = 30.0

#     delta = torch.clamp(delta, min=-delta_clip, max=delta_clip)
#     delta = delta * scale_factor

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     # ===== 更新 momentum =====
#     global_model.logits_momentum[class_id] = (
#         momentum_beta * global_model.logits_momentum[class_id] +
#         (1 - momentum_beta) * delta
#     )

#     # ===== Clip momentum norm =====
#     norm = global_model.logits_momentum[class_id].norm()
#     if norm > max_mom_norm:
#         global_model.logits_momentum[class_id] *= (max_mom_norm / norm)

#     # ===== α 線性成長控制 momentum 更新強度 =====
#     alpha = min(0.1 + 0.0032 * current_round, 0.67)

#     updated = (1 - alpha) * avg + alpha * (old_wb + global_model.logits_momentum[class_id])

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}

#     global_model.global_momentum[weight_key] = global_model.logits_momentum[class_id][:-1].detach().clone()
#     global_model.global_momentum[bias_key] = global_model.logits_momentum[class_id][-1].detach().clone()

#     print(f"[DEBUG] round={current_round:3d} | Δnorm={delta.norm():.4f} | mom_norm={norm:.4f} | α={alpha:.2f}")
#     return updated

#100Round femnist還可以
# def weighted_avg_with_momentum_ACG_v2(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)

#     avg = torch.sum(values * weights, dim=0)

#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     # lookahead baseline
#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id]).detach().clone().to(device)
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id]).detach().clone().to(device)
#         old_wb = torch.cat([lookahead_weight, lookahead_bias.unsqueeze(0)])
#     else:
#         old_wb = torch.cat([global_model.logits.weight[class_id], global_model.logits.bias[class_id].unsqueeze(0)])

#     delta = avg - old_wb

#     # 🔥 scale_factor動態變化，前期小後期稍大
#     scale_factor = min(0.2 + 0.0015 * current_round, 0.4)
#     delta = delta * scale_factor

#     # 🔥 delta clip，防止爆衝
#     delta_clip = 1.0
#     delta = torch.clamp(delta, min=-delta_clip, max=delta_clip)

#     # 🔥 adaptive momentum beta
#     if current_round < 100:
#         momentum_beta = 0.9
#     elif current_round < 200:
#         momentum_beta = 0.85
#     else:
#         momentum_beta = 0.8

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     # momentum update
#     global_model.logits_momentum[class_id] = (
#         momentum_beta * global_model.logits_momentum[class_id] +
#         (1 - momentum_beta) * delta
#     )

#     # ===== Momentum健康檢查與修正 =====
#     mom = global_model.logits_momentum[class_id]
#     mom_norm = mom.norm().item()

#     # clip norm
#     max_mom_norm = 15.0
#     if mom_norm > max_mom_norm:
#         mom = mom * (max_mom_norm / mom_norm)
#         global_model.logits_momentum[class_id] = mom

#     # cosine similarity檢查
#     cos_sim = torch.nn.functional.cosine_similarity(mom, delta, dim=0).item()
#     if cos_sim < -0.3:
#         print(f"[INFO] round={current_round} class_id={class_id} momentum reversed, shrinking")
#         global_model.logits_momentum[class_id] *= 0.5  # shrink

#     # 定期（每30輪）小幅shrinking（防止累積誤差）
#     if current_round > 0 and current_round % 30 == 0:
#         global_model.logits_momentum[class_id] *= 0.8

#     # 🔥 α 緩慢增長＋收斂
#     alpha = min(0.02 + 0.0015 * current_round, 0.25)  # 上限收斂到0.25

#     updated = (1 - alpha) * avg + alpha * (old_wb + global_model.logits_momentum[class_id])

#     # save updated global momentum
#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = global_model.logits_momentum[class_id][:-1].detach().clone()
#     global_model.global_momentum[bias_key] = global_model.logits_momentum[class_id][-1].detach().clone()

#     if current_round % 10 == 0:
#         print(f"[DEBUG] round={current_round:3d} | Δnorm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | α={alpha:.3f} | cos_sim={cos_sim:.3f}")

#     return updated

# import math

# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta_base: float = 0.9,
#     current_round: int = 0,
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)

#     avg = torch.sum(values * weights, dim=0)

#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(
#             weight_key, global_model.logits.weight[class_id]
#         ).clone().detach().to(device)
#         lookahead_bias = global_model.global_momentum.get(
#             bias_key, global_model.logits.bias[class_id]
#         ).clone().detach().to(device)
#         old_wb = torch.cat([lookahead_weight, lookahead_bias.unsqueeze(0)])
#     else:
#         old_wb = torch.cat([
#             global_model.logits.weight[class_id],
#             global_model.logits.bias[class_id].unsqueeze(0)
#         ])

#     delta = avg - old_wb
#     delta_norm = delta.norm().item()

#     if not hasattr(global_model, "stable_delta_norm") and current_round == 47:
#         global_model.stable_delta_norm = delta_norm
#         print(f"[INFO] 保存第47輪delta_norm={delta_norm:.4f}作為穩定標準")

#     if hasattr(global_model, "stable_delta_norm") and current_round > 47:
#         delta_norm = global_model.stable_delta_norm

#     if current_round < 20:
#         scale_factor = 0.2
#         delta_clip = 2.0
#     else:
#         scale_factor = min(0.2 + 0.001 * (current_round - 20), 0.35)
#         delta_clip = 2.0 + 0.005 * (current_round // 30)

#     if current_round >= 40:
#         scale_factor = min(scale_factor, 0.25)

#     if current_round >= 48:
#         delta_clip = 1.5
#         scale_factor = 0.15

#     delta = torch.clamp(delta, min=-delta_clip, max=delta_clip)
#     delta = delta * scale_factor

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)
#     mom = global_model.logits_momentum[class_id]

#     if mom.norm().item() == 0:
#         cos_sim = 1.0
#     else:
#         cos_sim = torch.nn.functional.cosine_similarity(mom, delta, dim=0).item()

#     delta_scale = math.tanh(0.5 * (delta_norm - 0.5))
#     cos_scale = (1 - cos_sim) / 2
#     adaptive_factor = 0.15 * delta_scale + 0.85 * cos_scale
#     momentum_beta = momentum_beta_base - 0.1 * adaptive_factor
#     momentum_beta = max(0.75, min(momentum_beta, 0.95))  # 保持自然限制，不強制拉高

#     new_mom = momentum_beta * mom + (1 - momentum_beta) * delta

#     if current_round >= 40:
#         new_mom = new_mom * 0.995

#     mom_norm = new_mom.norm().item()
#     max_mom_norm = 10.0 if current_round < 200 else 15.0
#     if current_round >= 40:
#         max_mom_norm = 8.0

#     if mom_norm > max_mom_norm:
#         new_mom *= (max_mom_norm / mom_norm)

#     if cos_sim < 0.0:
#         print(f"[SHRINK] round={current_round} | class_id={class_id} | cos_sim={cos_sim:.3f}")
#         new_mom *= 0.4

#     global_model.logits_momentum[class_id] = new_mom

#     trust = max(0.0, cos_sim) * (1.0 / (1.0 + delta_norm))
#     alpha = 0.15 + 0.3 * trust
#     alpha = min(alpha, 0.4)

#     if current_round >= 48:
#         alpha = min(alpha, 0.3)

#     updated = (1 - alpha) * avg + alpha * (old_wb + new_mom)

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}

#     global_model.global_momentum[weight_key] = new_mom[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_mom[-1].detach().clone()

#     if current_round % 10 == 0:
#         print(f"[DEBUG] round={current_round:3d} | Δnorm={delta_norm:.4f} | mom_norm={mom_norm:.4f} | α={alpha:.3f} | β={momentum_beta:.3f} | cos_sim={cos_sim:.3f}")

#     return updated

def weighted_avg_ACG(
    values: Union[list[torch.Tensor], torch.Tensor],
    weights: list[float],
    momentum: Union[torch.Tensor, None] = None
) -> torch.Tensor:
    """
    純粹做 weighted average，這裡可以考慮將動量納入加權平均
    """
    if isinstance(values, list):
        values = torch.stack(values)
    else:
        if values.dim() == 1:
            values = values.unsqueeze(0)

    weights = torch.tensor(weights, dtype=torch.float32, device=values.device)
    weights = weights / weights.sum()
    weights = weights.view(-1, 1)

    avg = torch.sum(values * weights, dim=0)
    
    # 如果有動量，則結合動量更新
    if momentum is not None:
        avg = avg + momentum

    return avg

# def weighted_avg_with_momentum_ACG - femnist準確度，100Round0.74(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device

#     # ===== 將輸入標準化為 [n, dim] =====
#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)

#     avg = torch.sum(values * weights, dim=0)
#     # ⭐ 不對 avg 做 clamp（保留更自然的訊號變化）
    
#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     # ===== 取出 lookahead baseline（從 global momentum 中提取） =====
#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(
#             weight_key, global_model.logits.weight[class_id]
#         ).clone().detach().to(device)

#         lookahead_bias = global_model.global_momentum.get(
#             bias_key, global_model.logits.bias[class_id]
#         ).clone().detach().to(device)

#         old_wb = torch.cat([lookahead_weight, lookahead_bias.unsqueeze(0)])
#     else:
#         old_wb = torch.cat([
#             global_model.logits.weight[class_id],
#             global_model.logits.bias[class_id].unsqueeze(0)
#         ])

#     delta = avg - old_wb
#     # ✅ 初期縮放避免爆衝，後期穩定強化
#     # if current_round < 100:
#     #     scale_factor = 0.3
#     #     delta_clip = 2.0
#     #     momentum_beta = 0.9
#     #     max_mom_norm = 20.0
#     # elif current_round < 200:
#     #     scale_factor = 0.5
#     #     delta_clip = 5.0
#     #     momentum_beta = 0.85
#     #     max_mom_norm = 30.0
#     # else:
#     #     scale_factor = 0.7
#     #     delta_clip = 6.0
#     #     momentum_beta = 0.8
#     #     max_mom_norm = 30.0
#     # delta_clip = 2.0  # 固定設置delta_clip
#     max_mom_norm = 15.0  # 設置更小的最大Momentum范數
#     delta_clip = 1.0  # 設置較小的delta_clip
#     # delta = delta * 0.2  # 更小的scale_factor

#     if current_round < 100:
#         momentum_beta = 0.9
#     elif current_round < 200:
#         momentum_beta = 0.85
#     else:
#         momentum_beta = 0.8


#     delta = torch.clamp(delta, min=-delta_clip, max=delta_clip)
#     delta = delta * 0.3 #scale_factor

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     # ===== 更新 momentum =====
#     global_model.logits_momentum[class_id] = (
#         momentum_beta * global_model.logits_momentum[class_id] +
#         (1 - momentum_beta) * delta
#     )

#     # ===== Clip momentum norm =====
#     norm = global_model.logits_momentum[class_id].norm()
#     if norm > max_mom_norm:
#         global_model.logits_momentum[class_id] *= (max_mom_norm / norm)

#     # ===== α 線性成長控制 momentum 更新強度 =====
#     alpha = min(0.1 + 0.0032 * current_round, 0.67)
#     # alpha = min(0.05 + 0.003 * current_round, 0.5)  # 減緩alpha的增長
#     # alpha = min(0.02 + 0.002 * current_round, 0.5)  # 更緩慢的增長


#     updated = (1 - alpha) * avg + alpha * (old_wb + global_model.logits_momentum[class_id])

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     if hasattr(global_model, "global_momentum"):
#         print("global_momentum 已初始化")
#     else:
#         print("global_momentum 尚未初始化")


#     global_model.global_momentum[weight_key] = global_model.logits_momentum[class_id][:-1].detach().clone()
#     global_model.global_momentum[bias_key] = global_model.logits_momentum[class_id][-1].detach().clone()

#     print(f"[DEBUG] round={current_round:3d} | Δnorm={delta.norm():.4f} | mom_norm={norm:.4f} | α={alpha:.2f}")
#     return updated


# import math

# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)

#     # 計算 weighted avg
#     avg = torch.sum(values * weights, dim=0)
#     avg = torch.clamp(avg, min=-50.0, max=50.0)

#     # 初始化 avg_history 結構
#     if not hasattr(global_model, "avg_history"):
#         global_model.avg_history = {}

#     prev_avg = global_model.avg_history.get(class_id, avg.clone())  # 若無歷史則直接使用當前 avg

#     # 動態平滑係數（可以根據輪數微調）
#     # 原本：
#     # ema_lambda = 0.8 if current_round < 50 else 0.9
#     # 改為更保守前期，開放後期
#     if current_round < 30:
#         ema_lambda = 0.85
#     elif current_round < 60:
#         ema_lambda = 0.88
#     else:
#         ema_lambda = 0.82  # 後期放鬆平滑，讓 avg 更具反應性


#     # 計算方向一致性（防止劇烈反轉時也平滑）
#     cos_sim_avg = torch.nn.functional.cosine_similarity(prev_avg, avg, dim=0).item()

#     # 印出平滑前的 avg norm 差異
#     print(f"[AVG-RAW] round={current_round:3d} | cos_sim={cos_sim_avg:.4f} | prev_avg.norm={prev_avg.norm():.2f} | raw_avg.norm={avg.norm():.2f}")

#     # 僅當方向一致性足夠時，才進行平滑
#     if cos_sim_avg > 0.2:
#         avg = ema_lambda * prev_avg + (1 - ema_lambda) * avg
#         print(f"[AVG-SMOOTHED] round={current_round:3d} | new_avg.norm={avg.norm():.2f}")
#     else:
#         print(f"[AVG-SKIP] round={current_round:3d} | cosine too low, skip smoothing")

#     # 儲存新的平滑 avg 供下輪使用
#     global_model.avg_history[class_id] = avg.detach().clone()


#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"
#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([
#             global_model.logits.weight[class_id],
#             global_model.logits.bias[class_id].unsqueeze(0)
#         ])

#     delta = avg - old_wb
#     delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
#     delta_norm = delta.norm().item()

#     if not hasattr(global_model, "prev_delta_norms"):
#         global_model.prev_delta_norms = {}
#     prev_delta_norm = global_model.prev_delta_norms.get(class_id, 10.0)
#     max_allowed_delta_norm = prev_delta_norm + 30.0
#     if delta_norm > max_allowed_delta_norm:
#         scale = max_allowed_delta_norm / delta_norm
#         delta *= scale
#         print(f"[SOFT-CLIP] Δnorm spike: {delta_norm:.2f} → {max_allowed_delta_norm:.2f} by scaling")
#     global_model.prev_delta_norms[class_id] = delta.norm().item()

#     if current_round < 20:
#         scale_factor = 0.3
#         delta_clip = 1.5
#         max_mom_norm = 15.0
#     elif current_round < 40:
#         scale_factor = 0.15
#         delta_clip = 1.0
#         max_mom_norm = 12.0
#     elif current_round < 70:
#         scale_factor = 0.11
#         delta_clip = 0.7
#         max_mom_norm = 11.0
#     else:
#         scale_factor = 0.10
#         delta_clip = 0.65
#         max_mom_norm = 13.0

#     delta = torch.clamp(delta, min=-delta_clip, max=delta_clip)
#     delta = delta * scale_factor

#     delta_norm_current = delta.norm().item()
#     delta_norm_cap = 150.0
#     if delta_norm_current > delta_norm_cap:
#         delta = delta * (delta_norm_cap / delta_norm_current)

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     cos_sim = torch.nn.functional.cosine_similarity(prev_momentum, delta, dim=0)
#     cos_sim_val = cos_sim.item()

#     if not hasattr(global_model, "momentum_flags"):
#         global_model.momentum_flags = {}
#     flag_count = global_model.momentum_flags.get(class_id, 0)

#     if cos_sim_val < 0.3:
#         flag_count += 1
#         if current_round < 20 and flag_count >= 1:
#             prev_momentum.zero_()
#             flag_count = 0
#         elif current_round < 50 and flag_count >= 2:
#             prev_momentum.zero_()
#             flag_count = 0
#         elif flag_count >= 2:
#             prev_momentum.zero_()
#             flag_count = 0
#     global_model.momentum_flags[class_id] = flag_count

#     if cos_sim_val < -0.1:
#         print(f"[CORRECT] cos_sim < -0.1, reversing delta direction")
#         delta = -delta
#         cos_sim_val = -cos_sim_val

#     if not hasattr(global_model, "momentum_variance"):
#         global_model.momentum_variance = {}
#     momentum_var = global_model.momentum_variance.get(class_id, 0.0)
#     momentum_var = 0.9 * momentum_var + 0.1 * (prev_momentum.norm().item() ** 2)
#     global_model.momentum_variance[class_id] = momentum_var

#     alpha = dynamic_alpha(cos_sim, delta_norm, current_round, mom_norm=prev_momentum.norm().item())
#     if momentum_var > 30.0:
#         alpha *= 0.7
#     if current_round < 30:
#         decay_lambda = 0.999
#     else:
#         decay_lambda = 0.993

#     new_momentum = decay_lambda * (
#         momentum_beta * prev_momentum + (1 - momentum_beta) * delta
#     )
#     new_momentum = torch.nan_to_num(new_momentum, nan=0.0, posinf=0.0, neginf=0.0)
#     mom_norm = new_momentum.norm()
#     if mom_norm > max_mom_norm:
#         new_momentum = new_momentum * (max_mom_norm / mom_norm)

#     global_model.logits_momentum[class_id] = new_momentum

#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     print(f"[DEBUG] round={current_round:3d} | Δnorm={delta_norm:.4f} | mom_norm={mom_norm:.4f} | α={alpha:.3f} | cos_sim={cos_sim:.4f}")
#     return updated

# # 調整 dynamic_alpha
# def dynamic_alpha(cos_sim: float, delta_norm: float, round: int, mom_norm: float = None) -> float:
#     if round < 30:
#         alpha_scale = 0.25
#         alpha_max = 0.4
#     elif round < 50:
#         alpha_scale = 0.18
#         alpha_max = 0.3
#     elif round < 70:
#         alpha_scale = 0.15
#         alpha_max = 0.28  # 後期略放寬
#     else:
#         alpha_scale = 0.14
#         alpha_max = 0.32  # 原為0.22，放寬至0.32

#     safe_norm = torch.tensor(delta_norm).clamp(min=1.0).log().item() / math.log(50.0)
#     norm_factor = 1 / (1 + math.exp(-safe_norm + 1.0))  # 原為+1.5，降低門檻

#     raw_alpha = cos_sim * norm_factor
#     if mom_norm is not None:
#         mom_factor = math.tanh(mom_norm / 8.0)
#         raw_alpha *= (1 + 0.25 * mom_factor)

#     alpha = alpha_scale * raw_alpha
#     return min(max(alpha, 0.0), alpha_max)

#femnist-0.76準確度，C16、E1
# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device

#     if isinstance(values, list):
#         # [DEBUG] client norm logging
#         for i, v in enumerate(values):
#             print(f"[client-{i}] round={current_round} | logits.norm={v.norm().item():.4f}")
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)
#         print(f"[single client] round={current_round} | logits.norm={values.norm().item():.4f}")

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)

#     avg = torch.sum(values * weights, dim=0)

#     # 新增：avg.norm clip，避免過大更新
#     if current_round < 40:
#         max_avg_norm = 10.0
#     elif current_round < 80:
#         max_avg_norm = 15.0
#     else:
#         max_avg_norm = 20.0
#     avg_norm = avg.norm().item()
#     if avg_norm > max_avg_norm:
#         print(f"[CLIP] round={current_round} | avg.norm={avg_norm:.4f} > {max_avg_norm}, applying clip.")
#         avg = avg * (max_avg_norm / avg.norm())

#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([
#             global_model.logits.weight[class_id],
#             global_model.logits.bias[class_id].unsqueeze(0)
#         ])

#     delta = avg - old_wb

#     # ========== 動態控制策略 ==========
#     if current_round < 100:
#         scale_factor = 0.3
#         delta_clip = 2.0
#         momentum_beta = 0.9
#         max_mom_norm = 20.0
#     elif current_round < 200:
#         scale_factor = 0.5
#         delta_clip = 5.0
#         momentum_beta = 0.85
#         max_mom_norm = 30.0
#     else:
#         scale_factor = 0.7
#         delta_clip = 6.0
#         momentum_beta = 0.8
#         max_mom_norm = 30.0

#     delta = torch.clamp(delta, min=-delta_clip, max=delta_clip)
#     delta = delta * scale_factor

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta

#     # Clip momentum norm
#     norm = new_momentum.norm()
#     if norm > max_mom_norm:
#         new_momentum = new_momentum * (max_mom_norm / norm)

#     global_model.logits_momentum[class_id] = new_momentum

#     # 線性成長 alpha 機制
#     alpha = min(0.1 + 0.0032 * current_round, 0.67)

#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     # 更新全域動量
#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     print(f"[DEBUG] round={current_round:3d} | avg.norm={avg.norm():.4f} | Δnorm={delta.norm():.4f} | mom_norm={norm:.4f} | α={alpha:.2f}")

#     return updated

# import torch
# import numpy as np
# from typing import Union

# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device

#     if isinstance(values, list):
#         norm_list = []
#         for i, v in enumerate(values):
#             norm_val = v.norm().item()
#             norm_list.append(norm_val)
#             print(f"[client-{i}] round={current_round} | logits.norm={norm_val:.4f}")
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)
#         norm_val = values.norm().item()
#         norm_list = [norm_val]
#         print(f"[single client] round={current_round} | logits.norm={norm_val:.4f}")

#     # --- Step 1: 客戶端輸入 logits norm 限制 ---
#     max_norm_allowed = 100.0
#     clipped_values = []
#     for i, v in enumerate(values):
#         norm_v = v.norm().item()
#         if not torch.isfinite(v).all() or norm_v > max_norm_allowed:
#             print(f"[WARN] client-{i} norm={norm_v:.4f}, clipping to max={max_norm_allowed}")
#             v = v * (max_norm_allowed / (norm_v + 1e-6))
#         clipped_values.append(v)
#     values = torch.stack(clipped_values)

#     # --- Step 2: 檢查 inf / nan ---
#     finite_mask = torch.isfinite(values).all(dim=1)
#     if not finite_mask.all():
#         print(f"[WARN] round={current_round} | Filtering {len(values) - finite_mask.sum().item()} non-finite logits")
#         values = values[finite_mask]
#         weights = torch.tensor(weights, dtype=torch.float32, device=device)[finite_mask]
#         if values.shape[0] == 0:
#             print("[ERROR] All client values are non-finite! Fallback to last global state.")
#             return global_model.logits.weight[class_id].detach().clone()
#     else:
#         weights = torch.tensor(weights, dtype=torch.float32, device=device)

#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)
#     avg = torch.sum(values * weights, dim=0)

#     # --- Step 3: 自適應 norm clip ---
#     if not hasattr(global_model, "norm_tracking"):
#         global_model.norm_tracking = {}

#     norm_array = np.array(norm_list)
#     norm_mean = norm_array.mean()
#     norm_std = norm_array.std()
#     prev_avg_norm = global_model.norm_tracking.get(class_id, norm_mean)

#     # 冷卻機制
#     if norm_std > 5.0:
#         max_avg_norm = min(norm_mean, 30.0)
#     else:
#         max_avg_norm = min(norm_mean + 2.0 * norm_std, 35.0, prev_avg_norm + 3.0)

#     avg_norm = avg.norm().item()
#     global_model.norm_tracking[class_id] = avg_norm

#     if avg_norm > max_avg_norm:
#         print(f"[CLIP] round={current_round} | avg.norm={avg_norm:.4f} > {max_avg_norm:.4f}, applying clip.")
#         avg = avg * (max_avg_norm / avg.norm())

#     # --- Step 4: 防止 inf 傳播 ---
#     if not torch.isfinite(avg).all():
#         print(f"[ERROR] round={current_round} | avg contains inf/nan. Using fallback.")
#         return global_model.logits.weight[class_id].detach().clone()

#     # --- Momentum 更新 ---
#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([
#             global_model.logits.weight[class_id],
#             global_model.logits.bias[class_id].unsqueeze(0)
#         ])

#     delta = avg - old_wb

#     # --- Step 5: 動態調參 ---
#     if current_round < 60:
#         scale_factor = 0.3
#         delta_clip = 2.0
#         momentum_beta = 0.9
#         max_mom_norm = 20.0
#         alpha_max = 0.67
#     elif current_round < 100:
#         scale_factor = 0.6
#         delta_clip = 5.0
#         momentum_beta = 0.85
#         max_mom_norm = 30.0
#         alpha_max = 0.8
#     else:
#         scale_factor = 0.8
#         delta_clip = 6.0
#         momentum_beta = 0.8
#         max_mom_norm = 40.0
#         alpha_max = 0.95

#     delta = torch.clamp(delta, min=-delta_clip, max=delta_clip)
#     delta = delta * scale_factor

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta

#     mom_norm = new_momentum.norm()
#     if mom_norm > max_mom_norm:
#         new_momentum = new_momentum * (max_mom_norm / mom_norm)

#     global_model.logits_momentum[class_id] = new_momentum

#     delta_norm = delta.norm().item()
#     if current_round < 60:
#         alpha_raw = min(0.1 + 0.0032 * current_round, alpha_max)
#     else:
#         alpha_raw = min(0.3 + 0.0045 * (current_round - 60), alpha_max)

#     if delta_norm <= 1.0:
#         stability_factor = 1.0
#     elif delta_norm >= 3.0:
#         stability_factor = 0.3
#     else:
#         stability_factor = 1.0 - ((delta_norm - 1.0) / 2.0) * (1.0 - 0.3)

#     alpha = max(alpha_raw * stability_factor, 0.2)

#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     print(f"[DEBUG] round={current_round:3d} | avg.norm={avg.norm():.4f} | Δnorm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | α={alpha:.2f}")

#     return updated

#- femnist前期收斂加速版
# def weighted_avg_with_momentum_ACG (
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device

#     if isinstance(values, list):
#         for i, v in enumerate(values):
#             print(f"[client-{i}] round={current_round} | logits.norm={v.norm().item():.4f}")
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)
#         print(f"[single client] round={current_round} | logits.norm={values.norm().item():.4f}")

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)

#     avg = torch.sum(values * weights, dim=0)

#     # -------- avg.norm clip --------
#     if current_round < 40:
#         max_avg_norm = 10.0
#     elif current_round < 80:
#         max_avg_norm = 15.0
#     else:
#         max_avg_norm = 20.0
#     avg_norm = avg.norm().item()
#     if avg_norm > max_avg_norm:
#         print(f"[CLIP] round={current_round} | avg.norm={avg_norm:.4f} > {max_avg_norm}, applying clip.")
#         avg = avg * (max_avg_norm / avg.norm())

#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([
#             global_model.logits.weight[class_id],
#             global_model.logits.bias[class_id].unsqueeze(0)
#         ])

#     delta = avg - old_wb

#     # -------- 動態控制策略 --------
#     if current_round < 100:
#         scale_factor = 0.3
#         delta_clip = 2.0
#         momentum_beta = 0.9
#         max_mom_norm = 20.0
#     elif current_round < 200:
#         scale_factor = 0.5
#         delta_clip = 5.0
#         momentum_beta = 0.85
#         max_mom_norm = 30.0
#     else:
#         scale_factor = 0.7
#         delta_clip = max(5.0 - 0.05 * (current_round - 200), 3.0)
#         momentum_beta = max(0.8 - 0.001 * (current_round - 200), 0.7)
#         max_mom_norm = 30.0

#     delta = torch.clamp(delta, min=-delta_clip, max=delta_clip)
#     delta = delta * scale_factor

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta

#     # Momentum 衰減（只在後期啟用）
#     if current_round >= 80:
#         decay_factor = 0.99 ** (current_round - 80)
#         new_momentum = new_momentum * decay_factor

#     # Clip momentum norm
#     norm = new_momentum.norm()
#     if norm > max_mom_norm:
#         new_momentum = new_momentum * (max_mom_norm / norm)

#     global_model.logits_momentum[class_id] = new_momentum

#     # -------- α 成長機制 --------
#     if current_round < 60:
#         alpha = min(0.1 + 0.0032 * current_round, 0.67)
#     elif current_round < 80:
#         alpha = min(0.3 + 0.005 * (current_round - 60), 0.8)
#     else:
#         alpha = min(0.4 + 0.002 * (current_round - 80), 0.85)

#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     print(f"[DEBUG] round={current_round:3d} | avg.norm={avg.norm():.4f} | Δnorm={delta.norm():.4f} | mom_norm={norm:.4f} | α={alpha:.2f}")

#     return updated

# femnist 前期快速攀升版本
# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device

#     if isinstance(values, list):
#         for i, v in enumerate(values):
#             print(f"[client-{i}] round={current_round} | logits.norm={v.norm().item():.4f}")
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)
#         print(f"[single client] round={current_round} | logits.norm={values.norm().item():.4f}")

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)

#     avg = torch.sum(values * weights, dim=0)

#     # -------- avg.norm clip --------
#     # 動態調整 avg.norm clip 的範圍
#     if current_round < 40:
#         max_avg_norm = 10.0
#     elif current_round < 60:
#         max_avg_norm = 15.0
#     elif current_round < 80:
#         max_avg_norm = 20.0
#     else:
#         max_avg_norm = 25.0
#     avg_norm = avg.norm().item()
#     if avg_norm > max_avg_norm:
#         print(f"[CLIP] round={current_round} | avg.norm={avg_norm:.4f} > {max_avg_norm}, applying clip.")
#         avg = avg * (max_avg_norm / avg.norm())

#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([
#             global_model.logits.weight[class_id],
#             global_model.logits.bias[class_id].unsqueeze(0)
#         ])

#     delta = avg - old_wb

#     # -------- 動態控制策略 --------
#     # 動態調整策略參數
#     scale_factor = min(0.3 + 0.004 * current_round, 0.7)  # 使 scale_factor 隨回合數動態調整
#     delta_clip = min(2.0 + 0.03 * current_round, 5.0)  # 動態調整 delta_clip
#     momentum_beta = max(0.8 - 0.001 * current_round, 0.75)
#     max_mom_norm = 20.0 + 0.05 * current_round  # 動態調整 max_mom_norm

#     delta = torch.clamp(delta, min=-delta_clip, max=delta_clip)
#     delta = delta * scale_factor

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta

#     # Momentum 衰減（只在後期啟用）
#     # Momentum 衰減（延後啟動，且衰減更平滑）
#     if current_round >= 200:
#         decay_factor = 0.999 ** (current_round - 200)
#         new_momentum = new_momentum * decay_factor

#     # Clip momentum norm
#     norm = new_momentum.norm()
#     if norm > max_mom_norm:
#         new_momentum = new_momentum * (max_mom_norm / norm)

    

#     # Clip momentum norm
#     norm = new_momentum.norm()
#     if norm > max_mom_norm:
#         new_momentum = new_momentum * (max_mom_norm / norm)

#     global_model.logits_momentum[class_id] = new_momentum

#     # -------- α 成長機制 --------
#     # 動態調整 α 成長機制
#     alpha = min(0.1 + 0.0032 * current_round, 0.67) if current_round < 60 else \
#             min(0.3 + 0.005 * (current_round - 60), 0.8) if current_round < 80 else \
#             min(0.4 + 0.002 * (current_round - 80), 0.85)

#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     print(f"[DEBUG] round={current_round:3d} | avg.norm={avg.norm():.4f} | Δnorm={delta.norm():.4f} | mom_norm={norm:.4f} | α={alpha:.2f}")

#     return updated

# 自適應avg.norm 版本
# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device

#     if isinstance(values, list):
#         for i, v in enumerate(values):
#             print(f"[client-{i}] round={current_round} | logits.norm={v.norm().item():.4f}")
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)
#         print(f"[single client] round={current_round} | logits.norm={values.norm().item():.4f}")

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)

#     avg = torch.sum(values * weights, dim=0)

#     # -------- avg.norm clip --------
#     # 動態調整 avg.norm clip 的範圍，根據歷史 norm 變化自適應
#     avg_norm = avg.norm().item()

#     # 計算當前範圍的最大最小值（可以是過去幾輪的均值或動態範圍）
#     if not hasattr(global_model, "history_norms"):
#         global_model.history_norms = []

#     # 只保留最近 n 輪的 norm 值
#     history_size = 30  # 可以設置為要保留的歷史回合數量
#     global_model.history_norms.append(avg_norm)
#     if len(global_model.history_norms) > history_size:
#         global_model.history_norms.pop(0)

#     # 基於歷史範圍調整 max_avg_norm
#     min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
#     max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0

#     # 根據訓練過程進行自適應調整
#     dynamic_range = max_avg_norm - min_avg_norm
#     adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

#     # 剪裁 avg.norm
#     if avg_norm > adjusted_max_avg_norm:
#         print(f"[CLIP] round={current_round} | avg.norm={avg_norm:.4f} > {adjusted_max_avg_norm:.4f}, applying clip.")
#         avg = avg * (adjusted_max_avg_norm / avg.norm())


#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([
#             global_model.logits.weight[class_id],
#             global_model.logits.bias[class_id].unsqueeze(0)
#         ])

#     delta = avg - old_wb

#     # -------- 動態控制策略 --------
#     # 動態調整策略參數
#     scale_factor = min(0.3 + 0.004 * current_round, 0.7)  # 使 scale_factor 隨回合數動態調整
#     delta_clip = min(2.0 + 0.03 * current_round, 5.0)  # 動態調整 delta_clip
#     momentum_beta = max(0.8 - 0.001 * current_round, 0.7)  # 動態調整 momentum_beta
#     max_mom_norm = 20.0 + 0.05 * current_round  # 動態調整 max_mom_norm

#     delta = torch.clamp(delta, min=-delta_clip, max=delta_clip)
#     delta = delta * scale_factor

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta

#     # Momentum 衰減（只在後期啟用）
#     if current_round >= 80:
#         decay_factor = 0.99 ** (current_round - 80)
#         new_momentum = new_momentum * decay_factor

#     # Clip momentum norm
#     norm = new_momentum.norm()
#     if norm > max_mom_norm:
#         new_momentum = new_momentum * (max_mom_norm / norm)

#     global_model.logits_momentum[class_id] = new_momentum

#     # -------- α 成長機制 --------
#     # 動態調整 α 成長機制
#     alpha = min(0.1 + 0.0032 * current_round, 0.67) if current_round < 60 else \
#             min(0.3 + 0.005 * (current_round - 60), 0.8) if current_round < 80 else \
#             min(0.4 + 0.002 * (current_round - 80), 0.85)

#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     print(f"[DEBUG] round={current_round:3d} | avg.norm={avg.norm():.4f} | Δnorm={delta.norm():.4f} | mom_norm={norm:.4f} | α={alpha:.2f}")

#     return updated


#綜合版本-->celeba成效不錯
# import math
# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)

#     avg = torch.sum(values * weights, dim=0)

#     # === 強 clip 保護極端情況 ===
#     if avg.norm() > 100.0:
#         avg = avg * (100.0 / avg.norm())

#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     # === 取出 lookahead baseline（可從 momentum 或當前參數）===
#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([
#             global_model.logits.weight[class_id],
#             global_model.logits.bias[class_id].unsqueeze(0)
#         ])

#     delta = avg - old_wb

#     # === 動態調整超參數（階段式）===
#     if current_round < 100:
#         scale_factor = 0.3
#         delta_clip = 2.0
#         momentum_beta = 0.9
#         max_mom_norm = 20.0
#     elif current_round < 200:
#         scale_factor = 0.5
#         delta_clip = 4.0
#         momentum_beta = 0.85
#         max_mom_norm = 30.0
#     else:
#         scale_factor = 0.7
#         delta_clip = 6.0
#         momentum_beta = 0.8
#         max_mom_norm = 35.0

#     delta = torch.clamp(delta, -delta_clip, delta_clip) * scale_factor

#     # === 初始化 momentum（依 class_id）===
#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta

#     # === clip momentum.norm（防止發散）===
#     mom_norm = new_momentum.norm()
#     if mom_norm > max_mom_norm:
#         new_momentum = new_momentum * (max_mom_norm / mom_norm)

#     global_model.logits_momentum[class_id] = new_momentum
#     # === 計算 avg.norm，先賦值再記錄 ===
#     avg_norm = avg.norm().item()

#     if not hasattr(global_model, "history_norms"):
#         global_model.history_norms = []

#     global_model.history_norms.append(avg_norm)
#     if len(global_model.history_norms) > 30:
#         global_model.history_norms.pop(0)

#     min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
#     max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0

#     dynamic_range = max_avg_norm - min_avg_norm
#     adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

#     print(f"round={current_round} | avg.norm={avg_norm:.4f} 、 {adjusted_max_avg_norm:.4f}, applying clip.")
    
#     # 下方是剪裁avg_norm
#     # if avg_norm < 1e-3 or math.isnan(avg_norm):
#     #     print(f"[WARNING] round={current_round} class={class_id} | avg.norm={avg_norm:.6f} 無效，採用 fallback")
#     #     avg = old_wb.detach().clone()
#     #     avg_norm = avg.norm().item()  # 更新 fallback 後的 avg.norm

    
#     # === α 線性成長，整合 avg + momentum（雙路徑學習）===
#     alpha = min(0.1 + 0.0032 * current_round, 0.8)

#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)
    
#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}

#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     print(f"[DEBUG] round={current_round:3d} | Δnorm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | α={alpha:.2f}")
#     return updated

# #可在celeba、femnist共用，但是並不是最好的
# import math
# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)

#     avg = torch.sum(values * weights, dim=0)

#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     # === lookahead baseline（來自 global_momentum 或目前參數）===
#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([
#             global_model.logits.weight[class_id],
#             global_model.logits.bias[class_id].unsqueeze(0)
#         ])
#     # 設定平滑的標準
#     # === 平滑 clip avg.norm()，避免方向爆炸 ===
#     avg_norm_val = avg.norm()
#     norm_limit = 100.0
#     if avg_norm_val > norm_limit:
#         shrink_factor = norm_limit / (avg_norm_val + 1e-6)
#         avg = avg * shrink_factor + old_wb * (1 - shrink_factor)

#     avg_norm = avg.norm().item()

#     if not hasattr(global_model, "history_norms"):
#         global_model.history_norms = []
#     global_model.history_norms.append(avg_norm)
#     if len(global_model.history_norms) > 25:
#         global_model.history_norms.pop(0)

#     min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
#     max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0
#     dynamic_range = max_avg_norm - min_avg_norm
#     adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

#     print(f"round={current_round} | avg.norm={avg_norm:.4f} > {adjusted_max_avg_norm:.4f}, applying clip.")

#     # === Fallback 判斷 ===
#     if avg_norm < 1e-3 or math.isnan(avg_norm):
#         print(f"[WARNING] round={current_round} class={class_id} | avg.norm={avg_norm:.6f} 無效，採用 fallback")
#         avg = old_wb.detach().clone()
#         avg_norm = avg.norm().item()

#     # === delta計算 + clip（向量方式）===
#     delta = avg - old_wb

#     # 動態超參數設定
#     if current_round < 100:
#         scale_factor = 0.3
#         delta_clip = 2.0
#         momentum_beta = 0.9
#         max_mom_norm = 20.0
#     elif current_round < 200:
#         scale_factor = 0.5
#         delta_clip = 4.0
#         momentum_beta = 0.85
#         max_mom_norm = 30.0
#     else:
#         scale_factor = 0.7
#         delta_clip = 6.0
#         momentum_beta = 0.8
#         max_mom_norm = 35.0

#     # clip delta 的 vector norm，而非每個元素
#     delta_norm_val = delta.norm()
#     if delta_norm_val > delta_clip:
#         delta = delta * (delta_clip / (delta_norm_val + 1e-6))
#     delta = delta * scale_factor

#     # === Momentum update ===
#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta

#     # clip momentum 長度
#     mom_norm = new_momentum.norm()
#     if mom_norm > max_mom_norm:
#         new_momentum = new_momentum * (max_mom_norm / (mom_norm + 1e-6))

#     global_model.logits_momentum[class_id] = new_momentum

    
#     # === 動態 α 調整（根據 avg.norm 穩定度）===
#     avg_norm_ratio = min(1.0, adjusted_max_avg_norm / (avg_norm + 1e-6))
#     alpha_base = min(0.1 + 0.0032 * current_round, 0.8)
#     alpha = alpha_base * avg_norm_ratio

#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     # === 記錄 momentum 參數備用 ===
#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}

#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     print(f"[DEBUG] round={current_round:3d} | Δnorm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | α={alpha:.2f}")
#     return updated

#   新增每 70 round 強化衝刺機制 可在基本全部執行，包含C32E4，但C8E4不行
# import math
# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
#     boost_interval: int = 70,  # 每多少 round 觸發一次 boost
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)

#     avg = torch.sum(values * weights, dim=0)

#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([
#             global_model.logits.weight[class_id],
#             global_model.logits.bias[class_id].unsqueeze(0)
#         ])

#     avg_norm_val = avg.norm()
#     norm_limit = 100.0
#     if avg_norm_val > norm_limit:
#         shrink_factor = norm_limit / (avg_norm_val + 1e-6)
#         avg = avg * shrink_factor + old_wb * (1 - shrink_factor)

#     avg_norm = avg.norm().item()

#     if not hasattr(global_model, "history_norms"):
#         global_model.history_norms = []
#     global_model.history_norms.append(avg_norm)
#     if len(global_model.history_norms) > 15:
#         global_model.history_norms.pop(0)

#     min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
#     max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0
#     dynamic_range = max_avg_norm - min_avg_norm
#     adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

#     if avg_norm < 1e-3 or math.isnan(avg_norm):
#         print(f"[WARNING] round={current_round} class={class_id} | avg.norm={avg_norm:.6f} 無效，採用 fallback")
#         avg = old_wb.detach().clone()
#         avg_norm = avg.norm().item()
    
#     delta = avg - old_wb

#     # === 每 boost_interval 啟動一次 boost_mode ===
#     boost_mode = (current_round >= boost_interval) and (current_round % boost_interval == 0)

#     if boost_mode:
#         scale_factor = 0.9
#         delta_clip = 8.0
#         momentum_beta = 0.7
#         max_mom_norm = 50.0
#     elif current_round < 100:
#         scale_factor = 0.3
#         delta_clip = 2.0
#         momentum_beta = 0.9
#         max_mom_norm = 20.0
#     elif current_round < 200:
#         scale_factor = 0.5
#         delta_clip = 4.0
#         momentum_beta = 0.85
#         max_mom_norm = 30.0
#     else:
#         scale_factor = 0.7
#         delta_clip = 6.0
#         momentum_beta = 0.8
#         max_mom_norm = 35.0

#     delta_norm_val = delta.norm()
#     if delta_norm_val > delta_clip:
#         delta = delta * (delta_clip / (delta_norm_val + 1e-6))
#     delta = delta * scale_factor

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)
#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta

#     mom_norm = new_momentum.norm()
#     if mom_norm > max_mom_norm:
#         new_momentum = new_momentum * (max_mom_norm / (mom_norm + 1e-6))
#     global_model.logits_momentum[class_id] = new_momentum

#     # === alpha 調整 ===
#     if boost_mode:
#         alpha = 0.9  # boost round 強化更新
#     else:
#         avg_norm_ratio = min(1.0, adjusted_max_avg_norm / (avg_norm + 1e-6))
#         alpha_base = 1 / (1 + math.exp(-0.03 * (current_round - 60))) * 0.8
#         alpha = alpha_base * avg_norm_ratio

#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     print(f"[DEBUG] round={current_round:3d} | Δnorm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | α={alpha:.2f} | boost_mode={boost_mode} | avg.norm={avg_norm:.4f} | adjusted_max={adjusted_max_avg_norm:.4f}")
#     return updated

# import torch
# import math
# from typing import Union

# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
#     boost_interval: int = 70,
#     min_clients_threshold: int = 4,
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)
#     avg = torch.sum(values * weights, dim=0)

#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([
#             global_model.logits.weight[class_id],
#             global_model.logits.bias[class_id].unsqueeze(0)
#         ])

#     avg_norm_val = avg.norm()
#     norm_limit = 100.0
#     if avg_norm_val > norm_limit:
#         shrink_factor = norm_limit / (avg_norm_val + 1e-6)
#         avg = avg * shrink_factor + old_wb * (1 - shrink_factor)

#     avg_norm = avg.norm().item()

#     if not hasattr(global_model, "history_norms"):
#         global_model.history_norms = []
#     global_model.history_norms.append(avg_norm)
#     if len(global_model.history_norms) > 15:
#         global_model.history_norms.pop(0)

#     min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
#     max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0
#     dynamic_range = max_avg_norm - min_avg_norm
#     adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

#     # === Early-stage 保護: 若 client 太少，完全禁用 momentum ===
#     if values.size(0) < min_clients_threshold:
#         print(f"[INFO] round={current_round} class={class_id} | client 太少（{values.size(0)}），跳過 momentum")
#         return avg.detach().clone()

#     # === avg 出錯或無效 ===
#     if avg_norm < 1e-3 or math.isnan(avg_norm) or math.isinf(avg_norm):
#         print(f"[WARNING] round={current_round} class={class_id} | avg 無效 (norm={avg_norm:.6f})，fallback to old_wb 並移除 momentum")
#         if class_id in global_model.logits_momentum:
#             global_model.logits_momentum.pop(class_id)
#         return old_wb.detach().clone()

#     delta = avg - old_wb

#     # === delta 出錯，直接 fallback ===
#     if torch.any(torch.isnan(delta)) or torch.any(torch.isinf(delta)):
#         print(f"[WARNING] round={current_round} class={class_id} | delta 出錯（nan/inf），取消更新")
#         if class_id in global_model.logits_momentum:
#             global_model.logits_momentum.pop(class_id)
#         return old_wb.detach().clone()

#     # === boost_mode 計算 ===
#     boost_mode = (current_round >= boost_interval) and (current_round % boost_interval == 0)

#     # === 動量參數與剪裁 ===
#     if boost_mode:
#         scale_factor = 0.9
#         delta_clip = 8.0
#         momentum_beta = 0.7
#         max_mom_norm = 50.0
#     elif current_round < 100:
#         scale_factor = 0.3
#         delta_clip = 2.0
#         momentum_beta = 0.9
#         max_mom_norm = 20.0
#     elif current_round < 200:
#         scale_factor = 0.5
#         delta_clip = 4.0
#         momentum_beta = 0.85
#         max_mom_norm = 30.0
#     else:
#         scale_factor = 0.7
#         delta_clip = 6.0
#         momentum_beta = 0.8
#         max_mom_norm = 35.0

#     # delta clip
#     delta_norm_val = delta.norm()
#     if delta_norm_val > delta_clip:
#         delta = delta * (delta_clip / (delta_norm_val + 1e-6))
#     delta = delta * scale_factor

#     if not hasattr(global_model, "logits_momentum"):
#         global_model.logits_momentum = {}

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta

#     mom_norm = new_momentum.norm()
#     if torch.any(torch.isnan(new_momentum)) or torch.any(torch.isinf(new_momentum)):
#         print(f"[WARNING] round={current_round} class={class_id} | momentum 出錯，使用 prev")
#         new_momentum = prev_momentum.detach().clone()
#     elif mom_norm > max_mom_norm:
#         new_momentum = new_momentum * (max_mom_norm / (mom_norm + 1e-6))

#     global_model.logits_momentum[class_id] = new_momentum

#     # === α 調整 ===
#     if boost_mode:
#         alpha = 0.9
#     else:
#         avg_norm_ratio = min(1.0, adjusted_max_avg_norm / (avg_norm + 1e-6))
#         alpha_base = 1 / (1 + math.exp(-0.03 * (current_round - 60))) * 0.8
#         alpha = alpha_base * avg_norm_ratio

#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()
#         # === 檢查更新結果是否過度偏離 ===
#     dist_old_to_avg = (avg - old_wb).norm()
#     dist_old_to_updated = (updated - old_wb).norm()

#     # 根據 round 設定不同偏離容忍度
#     if current_round < 50:
#         ratio_thresh = 1.2
#     elif current_round < 100:
#         ratio_thresh = 1.5
#     else:
#         ratio_thresh = 2.0

#     if dist_old_to_updated > dist_old_to_avg * ratio_thresh:
#         print(f"[WARNING] round={current_round} class={class_id} | updated 偏離過大，使用 avg 取代")
#         updated = avg.detach().clone()

#     print(f"[DEBUG] round={current_round:3d} | clients={values.size(0)} | Δnorm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | α={alpha:.2f} | boost={boost_mode} | avg.norm={avg_norm:.4f} | adj_max={adjusted_max_avg_norm:.4f}")
#     return updated


# clients數量過少也可使用的，動量(數量過少會有動量退場時機)、E的數量太多會崩潰
# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
#     boost_interval: int = 70,
#     min_clients_threshold: int = 7,
#     momentum_exit_round: int = 125
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device
#     num_clients = len(weights)

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)
#     avg = torch.sum(values * weights, dim=0)

#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([global_model.logits.weight[class_id], global_model.logits.bias[class_id].unsqueeze(0)])

#     avg_norm_val = avg.norm()
#     norm_limit = 100.0
#     if avg_norm_val > norm_limit:
#         shrink_factor = norm_limit / (avg_norm_val + 1e-6)
#         avg = avg * shrink_factor + old_wb * (1 - shrink_factor)

#     avg_norm = avg.norm().item()

#     if not hasattr(global_model, "history_norms"):
#         global_model.history_norms = []
#     global_model.history_norms.append(avg_norm)
#     if len(global_model.history_norms) > 15:
#         global_model.history_norms.pop(0)

#     min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
#     max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0
#     dynamic_range = max_avg_norm - min_avg_norm
#     adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

#     if values.size(0) < min_clients_threshold:
#         print(f"[INFO] round={current_round} class={class_id} | client 太少（{values.size(0)}），跳過 momentum")
#         return avg.detach().clone()

#     if current_round >= momentum_exit_round and values.size(0) < min_clients_threshold:
#         print(f"[INFO] round={current_round} class={class_id} | client 太少且回合數過多，禁用動量")
#         return avg.detach().clone()

#     if avg_norm < 1e-3 or math.isnan(avg_norm) or math.isinf(avg_norm):
#         print(f"[WARNING] round={current_round} class={class_id} | avg 無效，fallback to old_wb")
#         return old_wb.detach().clone()

#     delta = avg - old_wb
#     if torch.any(torch.isnan(delta)) or torch.any(torch.isinf(delta)):
#         print(f"[WARNING] round={current_round} class={class_id} | delta 出錯，取消更新")
#         return old_wb.detach().clone()

#     # === Boost & Momentum 參數調整 ===
#     if current_round < 100:
#         scale_factor, delta_clip, momentum_beta, max_mom_norm = 0.3, 2.0, 0.9, 20.0
#     elif current_round < 200:
#         scale_factor, delta_clip, momentum_beta, max_mom_norm = 0.5, 4.0, 0.85, 30.0
#     else:
#         scale_factor, delta_clip, momentum_beta, max_mom_norm = 0.7, 6.0, 0.8, 35.0

#     # 根據 clients 數調整動量強度
#     client_scale_ratio = min(1.0, max(0.0, (num_clients - 2) / 18))
#     scale_factor = 0.1 + 0.6 * client_scale_ratio
#     momentum_beta = 0.95 - 0.15 * client_scale_ratio
#     delta_clip = 2.0 + 6.0 * client_scale_ratio
#     max_mom_norm = 10.0 + 40.0 * client_scale_ratio

#     delta_norm_val = delta.norm()
#     if delta_norm_val > delta_clip:
#         delta = delta * (delta_clip / (delta_norm_val + 1e-6))
#     delta = delta * scale_factor

#     if not hasattr(global_model, "logits_momentum"):
#         global_model.logits_momentum = {}

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta
#     mom_norm = new_momentum.norm()

#     if torch.any(torch.isnan(new_momentum)) or torch.any(torch.isinf(new_momentum)):
#         print(f"[WARNING] round={current_round} class={class_id} | momentum 出錯，使用 prev")
#         new_momentum = prev_momentum.detach().clone()
#     elif mom_norm > max_mom_norm:
#         new_momentum = new_momentum * (max_mom_norm / (mom_norm + 1e-6))

#     global_model.logits_momentum[class_id] = new_momentum

#     avg_norm_ratio = min(1.0, adjusted_max_avg_norm / (avg_norm + 1e-6))
#     alpha_base = 1 / (1 + math.exp(-0.03 * (current_round - 60))) * 0.8
#     alpha = alpha_base * avg_norm_ratio

#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     dist_old_to_avg = (avg - old_wb).norm()
#     dist_old_to_updated = (updated - old_wb).norm()
#     ratio_thresh = 2.0 if current_round >= 100 else 1.5

#     if dist_old_to_updated > dist_old_to_avg * ratio_thresh:
#         print(f"[WARNING] round={current_round} class={class_id} | updated 偏離過大，使用 avg 取代")
#         updated = avg.detach().clone()

#     # ========== 新增：panic rollback 回復機制 ==========
#     if not hasattr(global_model, "rolling_stats"):
#         global_model.rolling_stats = {
#             "acc_history": [],
#             "last_updated_class_params": {},
#             "last_valid_round": -1,
#             "panic_mode": False
#         }

#     global_model.rolling_stats["last_updated_class_params"][class_id] = old_wb.detach().clone()
#     global_model.rolling_stats["last_valid_round"] = current_round

#     if global_model.rolling_stats.get("panic_mode", False):
#         print(f"[PANIC] round={current_round} class={class_id} | 準確率崩潰，回復上次參數")
#         return global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())

#     print(f"[DEBUG] round={current_round:3d} | clients={values.size(0)} | Δnorm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | α={alpha:.2f} | avg.norm={avg_norm:.4f}")
#     return updated


# -------------------------
# 此為femnist C16 E4，情況好的程式碼、C8E4會崩潰 400Round崩潰；C32，E4 15Round就崩潰
#---------------------------------------
# import math
# from typing import Union
# import torch

# import math
# from typing import Union
# import torch

# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
#     boost_interval: int = 70,
#     min_clients_threshold: int = 7,
#     momentum_exit_round: int = 125,
#     local_epoch: int = 4  # 🔧 新增：傳入本地訓練次數
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device
#     num_clients = len(weights)

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)
#     avg = torch.sum(values * weights, dim=0)

#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([global_model.logits.weight[class_id], global_model.logits.bias[class_id].unsqueeze(0)])

#     avg_norm_val = avg.norm()
#     norm_limit = 100.0
#     if avg_norm_val > norm_limit:
#         shrink_factor = norm_limit / (avg_norm_val + 1e-6)
#         avg = avg * shrink_factor + old_wb * (1 - shrink_factor)

#         # === 新增：平滑 avg ===
#     if values.size(0) < min_clients_threshold and local_epoch >= 4:
#         if not hasattr(global_model, "history_avg"):
#             global_model.history_avg = {}
#         if class_id not in global_model.history_avg:
#             global_model.history_avg[class_id] = avg.detach().clone()
#         else:
#             prev_avg = global_model.history_avg[class_id]
#             diff = avg - prev_avg
#             diff_norm = diff.norm().item()
#             prev_norm = prev_avg.norm().item()
#             if prev_norm > 0 and diff_norm / prev_norm > 0.2:
#                 # 如果 avg 與前一輪的差異超過 20%，則進行平滑處理
#                 smoothing_factor = 0.5  # 可以根據需要調整
#                 avg = prev_avg + smoothing_factor * diff
#             global_model.history_avg[class_id] = avg.detach().clone()

#     avg_norm = avg.norm().item()

#     if not hasattr(global_model, "history_norms"):
#         global_model.history_norms = []
#     global_model.history_norms.append(avg_norm)
#     if len(global_model.history_norms) > 15:
#         global_model.history_norms.pop(0)

#     min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
#     max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0
#     dynamic_range = max_avg_norm - min_avg_norm
#     adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

#     if values.size(0) < min_clients_threshold:
#         print(f"[INFO] round={current_round} class={class_id} | client 太少（{values.size(0)}），跳過 momentum")
#         return avg.detach().clone()

#     if current_round >= momentum_exit_round and values.size(0) < min_clients_threshold:
#         print(f"[INFO] round={current_round} class={class_id} | client 太少且回合數過多，禁用動量")
#         return avg.detach().clone()

#     if avg_norm < 1e-3 or math.isnan(avg_norm) or math.isinf(avg_norm):
#         print(f"[WARNING] round={current_round} class={class_id} | avg 無效，fallback to old_wb")
#         return old_wb.detach().clone()

#     delta = avg - old_wb
#     if torch.any(torch.isnan(delta)) or torch.any(torch.isinf(delta)):
#         print(f"[WARNING] round={current_round} class={class_id} | delta 出錯，取消更新")
#         return old_wb.detach().clone()

#     # === 新增：追蹤 delta.norm 和 avg.norm 長期變化 ===
#     if not hasattr(global_model, "history_avg_norms"):
#         global_model.history_avg_norms = {}
#     if not hasattr(global_model, "history_delta_norms"):
#         global_model.history_delta_norms = {}

#     if class_id not in global_model.history_avg_norms:
#         global_model.history_avg_norms[class_id] = []
#     if class_id not in global_model.history_delta_norms:
#         global_model.history_delta_norms[class_id] = []

#     global_model.history_avg_norms[class_id].append(avg.norm().item())
#     global_model.history_delta_norms[class_id].append(delta.norm().item())

#     if len(global_model.history_avg_norms[class_id]) > 10:
#         global_model.history_avg_norms[class_id].pop(0)
#     if len(global_model.history_delta_norms[class_id]) > 10:
#         global_model.history_delta_norms[class_id].pop(0)

#     def is_stable(seq: list, eps=1e-3):
#         return len(seq) >= 5 and all(abs(seq[i] - seq[i - 1]) < eps for i in range(1, len(seq)))

#     if is_stable(global_model.history_avg_norms[class_id]) and is_stable(global_model.history_delta_norms[class_id]):
#         noise = torch.randn_like(avg) * 0.01
#         avg = avg + noise
#         print(f"[NOISE] round={current_round} class={class_id} | learning stagnation detected, noise injected.")
#         delta = avg - old_wb  # 重新計算 delta

#     # === Boost & Momentum 調整（依 E 與 round 調整） ===
#     if current_round < 100:
#         scale_factor, delta_clip, momentum_beta, max_mom_norm = 0.3, 2.0, 0.9, 20.0
#     elif current_round < 200:
#         scale_factor, delta_clip, momentum_beta, max_mom_norm = 0.5, 4.0, 0.85, 30.0
#     else:
#         scale_factor, delta_clip, momentum_beta, max_mom_norm = 0.7, 6.0, 0.8, 35.0

#     client_scale_ratio = min(1.0, max(0.0, (num_clients - 2) / 18))
#     scale_factor = 0.1 + 0.6 * client_scale_ratio
#     momentum_beta = 0.95 - 0.15 * client_scale_ratio
#     delta_clip = 2.0 + 6.0 * client_scale_ratio
#     max_mom_norm = 10.0 + 40.0 * client_scale_ratio

#     # === 根據 local epoch 調整 clip 強度 ===
#     local_epoch_scale = 1.0 / (1.0 + 0.2 * (local_epoch - 1))
#     delta_clip *= local_epoch_scale
#     scale_factor *= local_epoch_scale

#     delta_norm_val = delta.norm()
#     if delta_norm_val > delta_clip:
#         delta = delta * (delta_clip / (delta_norm_val + 1e-6))
#     delta = delta * scale_factor

#     if not hasattr(global_model, "logits_momentum"):
#         global_model.logits_momentum = {}

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta
#     mom_norm = new_momentum.norm()

#     if torch.any(torch.isnan(new_momentum)) or torch.any(torch.isinf(new_momentum)):
#         print(f"[WARNING] round={current_round} class={class_id} | momentum 出錯，使用 prev")
#         new_momentum = prev_momentum.detach().clone()
#     elif mom_norm > max_mom_norm:
#         new_momentum = new_momentum * (max_mom_norm / (mom_norm + 1e-6))

#     global_model.logits_momentum[class_id] = new_momentum

#     avg_norm_ratio = min(1.0, adjusted_max_avg_norm / (avg_norm + 1e-6))
#     alpha_base = 1 / (1 + math.exp(-0.03 * (current_round - 60))) * 0.8
#     alpha = alpha_base * avg_norm_ratio
#     print(f"[DEBUG] round={current_round} class={class_id} | Pre-update avg: {avg.detach().cpu().numpy()}")
#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     dist_old_to_avg = (avg - old_wb).norm()
#     dist_old_to_updated = (updated - old_wb).norm()
#     ratio_thresh = 2.0 if current_round >= 100 else 1.5

#     if dist_old_to_updated > dist_old_to_avg * ratio_thresh:
#         print(f"[WARNING] round={current_round} class={class_id} | updated 偏離過大，使用 avg 取代")
#         updated = avg.detach().clone()

#     if updated.norm().item() < 1e-4:
#         print(f"[WARNING] round={current_round} class={class_id} | updated.norm 太小 fallback to avg")
#         updated = avg.detach().clone()

#     if not hasattr(global_model, "rolling_stats"):
#         global_model.rolling_stats = {
#             "acc_history": [],
#             "last_updated_class_params": {},
#             "last_valid_round": -1,
#             "panic_mode": False
#         }

#     global_model.rolling_stats["last_updated_class_params"][class_id] = old_wb.detach().clone()
#     global_model.rolling_stats["last_valid_round"] = current_round

#     if global_model.rolling_stats.get("panic_mode", False):
#         print(f"[PANIC] round={current_round} class={class_id} | 準確率崩潰，回復上次參數")
#         return global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())

#     print(f"[DEBUG] round={current_round:3d} | clients={values.size(0)} | Δnorm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | α={alpha:.2f} | avg.norm={avg_norm:.4f}")
#     return updated

# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
#     boost_interval: int = 70,
#     min_clients_threshold: int = 7,
#     momentum_exit_round: int = 125,
#     local_epoch: int = 4,
#     total_rounds: int = 500  # 🔧 新增：傳入總輪數，用於後期平滑控制
# ) -> torch.Tensor:
#     import math
#     from collections import deque
#     device = global_model.logits.weight.device
#     num_clients = len(weights)

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)
#     avg = torch.sum(values * weights, dim=0)

#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([global_model.logits.weight[class_id], global_model.logits.bias[class_id].unsqueeze(0)])

#     avg_norm_val = avg.norm()
#     norm_limit = 100.0
#     if avg_norm_val > norm_limit:
#         shrink_factor = norm_limit / (avg_norm_val + 1e-6)
#         avg = avg * shrink_factor + old_wb * (1 - shrink_factor)

#     # === 新增：平滑 avg（早期） ===
#     if values.size(0) < min_clients_threshold and local_epoch >= 4:
#         if not hasattr(global_model, "history_avg"):
#             global_model.history_avg = {}
#         if class_id not in global_model.history_avg:
#             global_model.history_avg[class_id] = avg.detach().clone()
#         else:
#             prev_avg = global_model.history_avg[class_id]
#             diff = avg - prev_avg
#             diff_norm = diff.norm().item()
#             prev_norm = prev_avg.norm().item()
#             if prev_norm > 0 and diff_norm / prev_norm > 0.2:
#                 smoothing_factor = 0.5
#                 avg = prev_avg + smoothing_factor * diff
#             global_model.history_avg[class_id] = avg.detach().clone()

#     # === 🔻新增：後期平滑 avg 避免暴衝 ===
#     if not hasattr(global_model, "history_avg_buffer"):
#         global_model.history_avg_buffer = {}
#     if class_id not in global_model.history_avg_buffer:
#         global_model.history_avg_buffer[class_id] = deque(maxlen=5)

#     global_model.history_avg_buffer[class_id].append(avg.detach().clone())

#     smoothing_start_round = total_rounds - 5
#     if current_round >= smoothing_start_round:
#         buffer = global_model.history_avg_buffer[class_id]
#         if len(buffer) > 1:
#             smoothed_avg = sum(buffer) / len(buffer)
#             avg = smoothed_avg
#     # === 🔺新增結束 ===

#     avg_norm = avg.norm().item()

#     if not hasattr(global_model, "history_norms"):
#         global_model.history_norms = []
#     global_model.history_norms.append(avg_norm)
#     if len(global_model.history_norms) > 15:
#         global_model.history_norms.pop(0)

#     min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
#     max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0
#     dynamic_range = max_avg_norm - min_avg_norm
#     adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

#     if values.size(0) < min_clients_threshold:
#         print(f"[INFO] round={current_round} class={class_id} | client 太少（{values.size(0)}），跳過 momentum")
#         return avg.detach().clone()

#     if current_round >= momentum_exit_round and values.size(0) < min_clients_threshold:
#         print(f"[INFO] round={current_round} class={class_id} | client 太少且回合數過多，禁用動量")
#         return avg.detach().clone()

#     if avg_norm < 1e-3 or math.isnan(avg_norm) or math.isinf(avg_norm):
#         print(f"[WARNING] round={current_round} class={class_id} | avg 無效，fallback to old_wb")
#         return old_wb.detach().clone()

#     delta = avg - old_wb
#     if torch.any(torch.isnan(delta)) or torch.any(torch.isinf(delta)):
#         print(f"[WARNING] round={current_round} class={class_id} | delta 出錯，取消更新")
#         return old_wb.detach().clone()

#     if not hasattr(global_model, "history_avg_norms"):
#         global_model.history_avg_norms = {}
#     if not hasattr(global_model, "history_delta_norms"):
#         global_model.history_delta_norms = {}

#     if class_id not in global_model.history_avg_norms:
#         global_model.history_avg_norms[class_id] = []
#     if class_id not in global_model.history_delta_norms:
#         global_model.history_delta_norms[class_id] = []

#     global_model.history_avg_norms[class_id].append(avg.norm().item())
#     global_model.history_delta_norms[class_id].append(delta.norm().item())

#     if len(global_model.history_avg_norms[class_id]) > 10:
#         global_model.history_avg_norms[class_id].pop(0)
#     if len(global_model.history_delta_norms[class_id]) > 10:
#         global_model.history_delta_norms[class_id].pop(0)

#     def is_stable(seq: list, eps=1e-3):
#         return len(seq) >= 5 and all(abs(seq[i] - seq[i - 1]) < eps for i in range(1, len(seq)))

#     if is_stable(global_model.history_avg_norms[class_id]) and is_stable(global_model.history_delta_norms[class_id]):
#         noise = torch.randn_like(avg) * 0.01
#         avg = avg + noise
#         print(f"[NOISE] round={current_round} class={class_id} | learning stagnation detected, noise injected.")
#         delta = avg - old_wb

#     if current_round < 100:
#         scale_factor, delta_clip, momentum_beta, max_mom_norm = 0.3, 2.0, 0.9, 20.0
#     elif current_round < 200:
#         scale_factor, delta_clip, momentum_beta, max_mom_norm = 0.5, 4.0, 0.85, 30.0
#     else:
#         scale_factor, delta_clip, momentum_beta, max_mom_norm = 0.7, 6.0, 0.8, 35.0

#     client_scale_ratio = min(1.0, max(0.0, (num_clients - 2) / 18))
#     scale_factor = 0.1 + 0.6 * client_scale_ratio
#     momentum_beta = 0.95 - 0.15 * client_scale_ratio
#     delta_clip = 2.0 + 6.0 * client_scale_ratio
#     max_mom_norm = 10.0 + 40.0 * client_scale_ratio

#     local_epoch_scale = 1.0 / (1.0 + 0.2 * (local_epoch - 1))
#     delta_clip *= local_epoch_scale
#     scale_factor *= local_epoch_scale

#     delta_norm_val = delta.norm()
#     if delta_norm_val > delta_clip:
#         delta = delta * (delta_clip / (delta_norm_val + 1e-6))
#     delta = delta * scale_factor

#     if not hasattr(global_model, "logits_momentum"):
#         global_model.logits_momentum = {}

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta
#     mom_norm = new_momentum.norm()

#     if torch.any(torch.isnan(new_momentum)) or torch.any(torch.isinf(new_momentum)):
#         print(f"[WARNING] round={current_round} class={class_id} | momentum 出錯，使用 prev")
#         new_momentum = prev_momentum.detach().clone()
#     elif mom_norm > max_mom_norm:
#         new_momentum = new_momentum * (max_mom_norm / (mom_norm + 1e-6))

#     global_model.logits_momentum[class_id] = new_momentum

#     avg_norm_ratio = min(1.0, adjusted_max_avg_norm / (avg_norm + 1e-6))
#     alpha_base = 1 / (1 + math.exp(-0.03 * (current_round - 60))) * 0.8
#     alpha = alpha_base * avg_norm_ratio
#     print(f"[DEBUG] round={current_round} class={class_id} | Pre-update avg: {avg.detach().cpu().numpy()}")
#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     dist_old_to_avg = (avg - old_wb).norm()
#     dist_old_to_updated = (updated - old_wb).norm()
#     ratio_thresh = 2.0 if current_round >= 100 else 1.5

#     if dist_old_to_updated > dist_old_to_avg * ratio_thresh:
#         print(f"[WARNING] round={current_round} class={class_id} | updated 偏離過大，使用 avg 取代")
#         updated = avg.detach().clone()

#     if updated.norm().item() < 1e-4:
#         print(f"[WARNING] round={current_round} class={class_id} | updated.norm 太小 fallback to avg")
#         updated = avg.detach().clone()

#     if not hasattr(global_model, "rolling_stats"):
#         global_model.rolling_stats = {
#             "acc_history": [],
#             "last_updated_class_params": {},
#             "last_valid_round": -1,
#             "panic_mode": False
#         }

#     global_model.rolling_stats["last_updated_class_params"][class_id] = old_wb.detach().clone()
#     global_model.rolling_stats["last_valid_round"] = current_round

#     if global_model.rolling_stats.get("panic_mode", False):
#         print(f"[PANIC] round={current_round} class={class_id} | 準確率崩潰，回復上次參數")
#         return global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())

#     print(f"[DEBUG] round={current_round:3d} | clients={values.size(0)} | Δnorm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | α={alpha:.2f} | avg.norm={avg_norm:.4f}")
#     return updated


#-----------------------------
#16以上的Clinet沒問題，C8、E4會崩潰剩餘沒問題、287Round崩潰、C32E4崩潰
#-----------------------------
# import math
# from typing import Union
# import torch

# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
#     boost_interval: int = 70,
#     min_clients_threshold: int = 7,
#     momentum_exit_round: int = 125,
#     local_epoch: int = 4  # 🔧 新增：傳入本地訓練次數
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device
#     num_clients = len(weights)

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)
#     avg = torch.sum(values * weights, dim=0)

#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([global_model.logits.weight[class_id], global_model.logits.bias[class_id].unsqueeze(0)])

#     avg_norm_val = avg.norm()
#     norm_limit = 100.0
#     if avg_norm_val > norm_limit:
#         shrink_factor = norm_limit / (avg_norm_val + 1e-6)
#         avg = avg * shrink_factor + old_wb * (1 - shrink_factor)

#         # === 新增：平滑 avg ===
#     if values.size(0) < min_clients_threshold and local_epoch >= 4:
#         if not hasattr(global_model, "history_avg"):
#             global_model.history_avg = {}
#         if class_id not in global_model.history_avg:
#             global_model.history_avg[class_id] = avg.detach().clone()
#         else:
#             prev_avg = global_model.history_avg[class_id]
#             diff = avg - prev_avg
#             diff_norm = diff.norm().item()
#             prev_norm = prev_avg.norm().item()
#             if prev_norm > 0 and diff_norm / prev_norm > 0.2:
#                 # 如果 avg 與前一輪的差異超過 20%，則進行平滑處理
#                 smoothing_factor = 0.5  # 可以根據需要調整
#                 avg = prev_avg + smoothing_factor * diff
#             global_model.history_avg[class_id] = avg.detach().clone()

#     avg_norm = avg.norm().item()

#     if not hasattr(global_model, "history_norms"):
#         global_model.history_norms = []
#     global_model.history_norms.append(avg_norm)
#     if len(global_model.history_norms) > 15:
#         global_model.history_norms.pop(0)

#     min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
#     max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0
#     dynamic_range = max_avg_norm - min_avg_norm
#     adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

#     if values.size(0) < min_clients_threshold:
#         print(f"[INFO] round={current_round} class={class_id} | client 太少（{values.size(0)}），跳過 momentum")
#         return avg.detach().clone()

#     if current_round >= momentum_exit_round and values.size(0) < min_clients_threshold:
#         print(f"[INFO] round={current_round} class={class_id} | client 太少且回合數過多，禁用動量")
#         return avg.detach().clone()

#     if avg_norm < 1e-3 or math.isnan(avg_norm) or math.isinf(avg_norm):
#         print(f"[WARNING] round={current_round} class={class_id} | avg 無效，fallback to old_wb")
#         return old_wb.detach().clone()

#     delta = avg - old_wb
#     if torch.any(torch.isnan(delta)) or torch.any(torch.isinf(delta)):
#         print(f"[WARNING] round={current_round} class={class_id} | delta 出錯，取消更新")
#         return old_wb.detach().clone()

#     # === 新增：追蹤 delta.norm 和 avg.norm 長期變化 ===
#     if not hasattr(global_model, "history_avg_norms"):
#         global_model.history_avg_norms = {}
#     if not hasattr(global_model, "history_delta_norms"):
#         global_model.history_delta_norms = {}

#     if class_id not in global_model.history_avg_norms:
#         global_model.history_avg_norms[class_id] = []
#     if class_id not in global_model.history_delta_norms:
#         global_model.history_delta_norms[class_id] = []

#     global_model.history_avg_norms[class_id].append(avg.norm().item())
#     global_model.history_delta_norms[class_id].append(delta.norm().item())

#     if len(global_model.history_avg_norms[class_id]) > 10:
#         global_model.history_avg_norms[class_id].pop(0)
#     if len(global_model.history_delta_norms[class_id]) > 10:
#         global_model.history_delta_norms[class_id].pop(0)

#     def is_stable(seq: list, eps=1e-3):
#         return len(seq) >= 5 and all(abs(seq[i] - seq[i - 1]) < eps for i in range(1, len(seq)))

#     if is_stable(global_model.history_avg_norms[class_id]) and is_stable(global_model.history_delta_norms[class_id]):
#         noise = torch.randn_like(avg) * 0.01
#         avg = avg + noise
#         print(f"[NOISE] round={current_round} class={class_id} | learning stagnation detected, noise injected.")
#         delta = avg - old_wb  # 重新計算 delta

#     # === Boost & Momentum 調整（依 E 與 round 調整） ===
#     if current_round < 100:
#         scale_factor, delta_clip, momentum_beta, max_mom_norm = 0.3, 2.0, 0.9, 20.0
#     elif current_round < 200:
#         scale_factor, delta_clip, momentum_beta, max_mom_norm = 0.5, 4.0, 0.85, 30.0
#     else:
#         scale_factor, delta_clip, momentum_beta, max_mom_norm = 0.7, 6.0, 0.8, 35.0

#     client_scale_ratio = min(1.0, max(0.0, (num_clients - 2) / 18))
#     scale_factor = 0.1 + 0.6 * client_scale_ratio
#     momentum_beta = 0.95 - 0.15 * client_scale_ratio
#     delta_clip = 2.0 + 6.0 * client_scale_ratio
#     max_mom_norm = 10.0 + 40.0 * client_scale_ratio

#     # === 根據 local epoch 調整 clip 強度 ===
#     local_epoch_scale = 1.0 / (1.0 + 0.2 * (local_epoch - 1))
#     delta_clip *= local_epoch_scale
#     scale_factor *= local_epoch_scale

#     delta_norm_val = delta.norm()
#     if delta_norm_val > delta_clip:
#         delta = delta * (delta_clip / (delta_norm_val + 1e-6))
#     delta = delta * scale_factor

#     if not hasattr(global_model, "logits_momentum"):
#         global_model.logits_momentum = {}

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta
#     mom_norm = new_momentum.norm()

#     if torch.any(torch.isnan(new_momentum)) or torch.any(torch.isinf(new_momentum)):
#         print(f"[WARNING] round={current_round} class={class_id} | momentum 出錯，使用 prev")
#         new_momentum = prev_momentum.detach().clone()
#     elif mom_norm > max_mom_norm:
#         new_momentum = new_momentum * (max_mom_norm / (mom_norm + 1e-6))

#     global_model.logits_momentum[class_id] = new_momentum

#     avg_norm_ratio = min(1.0, adjusted_max_avg_norm / (avg_norm + 1e-6))
#     alpha_base = 1 / (1 + math.exp(-0.03 * (current_round - 60))) * 0.8
#     alpha = alpha_base * avg_norm_ratio
#     print(f"[DEBUG] round={current_round} class={class_id} | avg.norm={avg_norm:.4f}")
#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     dist_old_to_avg = (avg - old_wb).norm()
#     dist_old_to_updated = (updated - old_wb).norm()
#     ratio_thresh = 2.0 if current_round >= 100 else 1.5

#     if dist_old_to_updated > dist_old_to_avg * ratio_thresh:
#         print(f"[WARNING] round={current_round} class={class_id} | updated 偏離過大，使用 avg 取代")
#         updated = avg.detach().clone()

#     if updated.norm().item() < 1e-4:
#         print(f"[WARNING] round={current_round} class={class_id} | updated.norm 太小 fallback to avg")
#         updated = avg.detach().clone()

#     if not hasattr(global_model, "rolling_stats"):
#         global_model.rolling_stats = {
#             "acc_history": [],
#             "last_updated_class_params": {},
#             "last_valid_round": -1,
#             "panic_mode": False
#         }

#     global_model.rolling_stats["last_updated_class_params"][class_id] = old_wb.detach().clone()
#     global_model.rolling_stats["last_valid_round"] = current_round

#     if global_model.rolling_stats.get("panic_mode", False):
#         print(f"[PANIC] round={current_round} class={class_id} | 準確率崩潰，回復上次參數")
#         return global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())

#     print(f"[DEBUG] round={current_round:3d} | clients={values.size(0)} | Δnorm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | α={alpha:.2f} | avg.norm={avg_norm:.4f}")
#     return updated


#198Round C8E4崩潰
# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
#     boost_interval: int = 70,
#     min_clients_threshold: int = 7,
#     momentum_exit_round: int = 125,
#     local_epoch: int = 4
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device
#     num_clients = len(weights)

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)
#     avg = torch.sum(values * weights, dim=0)

#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([global_model.logits.weight[class_id], global_model.logits.bias[class_id].unsqueeze(0)])

#     avg_norm_val = avg.norm()
#     norm_limit = 100.0
#     if avg_norm_val > norm_limit:
#         shrink_factor = norm_limit / (avg_norm_val + 1e-6)
#         avg = avg * shrink_factor + old_wb * (1 - shrink_factor)

#     if values.size(0) < min_clients_threshold and local_epoch >= 4:
#         if not hasattr(global_model, "history_avg"):
#             global_model.history_avg = {}
#         if class_id not in global_model.history_avg:
#             global_model.history_avg[class_id] = avg.detach().clone()
#         else:
#             prev_avg = global_model.history_avg[class_id]
#             diff = avg - prev_avg
#             diff_norm = diff.norm().item()
#             prev_norm = prev_avg.norm().item()
#             if prev_norm > 0 and diff_norm / prev_norm > 0.2:
#                 smoothing_factor = 0.5
#                 avg = prev_avg + smoothing_factor * diff
#             global_model.history_avg[class_id] = avg.detach().clone()

#     avg_norm = avg.norm().item()

#     if not hasattr(global_model, "history_norms"):
#         global_model.history_norms = []
#     global_model.history_norms.append(avg_norm)
#     if len(global_model.history_norms) > 15:
#         global_model.history_norms.pop(0)

#     min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
#     max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0
#     dynamic_range = max_avg_norm - min_avg_norm
#     adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

#     early_stage_momentum = (
#         values.size(0) < min_clients_threshold and
#         current_round < momentum_exit_round and
#         local_epoch >= 4
#     )

#     if values.size(0) < min_clients_threshold and not early_stage_momentum:
#         print(f"[INFO] round={current_round} class={class_id} | client 太少（{values.size(0)}），跳過 momentum")
#         return avg.detach().clone()

#     if avg_norm < 1e-3 or math.isnan(avg_norm) or math.isinf(avg_norm):
#         print(f"[WARNING] round={current_round} class={class_id} | avg 無效，fallback to old_wb")
#         return old_wb.detach().clone()

#     delta = avg - old_wb
#     if torch.any(torch.isnan(delta)) or torch.any(torch.isinf(delta)):
#         print(f"[WARNING] round={current_round} class={class_id} | delta 出錯，取消更新")
#         return old_wb.detach().clone()

#     if not hasattr(global_model, "history_avg_norms"):
#         global_model.history_avg_norms = {}
#     if not hasattr(global_model, "history_delta_norms"):
#         global_model.history_delta_norms = {}

#     if class_id not in global_model.history_avg_norms:
#         global_model.history_avg_norms[class_id] = []
#     if class_id not in global_model.history_delta_norms:
#         global_model.history_delta_norms[class_id] = []

#     global_model.history_avg_norms[class_id].append(avg.norm().item())
#     global_model.history_delta_norms[class_id].append(delta.norm().item())

#     if len(global_model.history_avg_norms[class_id]) > 10:
#         global_model.history_avg_norms[class_id].pop(0)
#     if len(global_model.history_delta_norms[class_id]) > 10:
#         global_model.history_delta_norms[class_id].pop(0)

#     def is_stable(seq: list, eps=1e-3):
#         return len(seq) >= 5 and all(abs(seq[i] - seq[i - 1]) < eps for i in range(1, len(seq)))

#     if is_stable(global_model.history_avg_norms[class_id]) and is_stable(global_model.history_delta_norms[class_id]):
#         noise = torch.randn_like(avg) * 0.01
#         avg = avg + noise
#         print(f"[NOISE] round={current_round} class={class_id} | learning stagnation detected, noise injected.")
#         delta = avg - old_wb

#     # if current_round < 100:
#     #     scale_factor, delta_clip, base_beta, max_mom_norm = 0.3, 2.0, 0.9, 20.0
#     # elif current_round < 200:
#     #     scale_factor, delta_clip, base_beta, max_mom_norm = 0.5, 4.0, 0.85, 30.0
#     # else:
#     #     scale_factor, delta_clip, base_beta, max_mom_norm = 0.7, 6.0, 0.8, 35.0

#     client_scale_ratio = min(1.0, max(0.0, (num_clients - 2) / 18))
#     scale_factor = 0.1 + 0.6 * client_scale_ratio
#     base_beta = 0.95 - 0.15 * client_scale_ratio
#     delta_clip = 2.0 + 6.0 * client_scale_ratio
#     max_mom_norm = 10.0 + 40.0 * client_scale_ratio

#     if early_stage_momentum:
#         momentum_beta = max(0.5, base_beta * (1.0 - current_round / momentum_exit_round))
#     else:
#         momentum_beta = base_beta

#     local_epoch_scale = 1.0 / (1.0 + 0.2 * (local_epoch - 1))
#     delta_clip *= local_epoch_scale
#     scale_factor *= local_epoch_scale

#     delta_norm_val = delta.norm()
#     if delta_norm_val > delta_clip:
#         delta = delta * (delta_clip / (delta_norm_val + 1e-6))
#     delta = delta * scale_factor

#     if not hasattr(global_model, "logits_momentum"):
#         global_model.logits_momentum = {}

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta
#     mom_norm = new_momentum.norm()

#     if torch.any(torch.isnan(new_momentum)) or torch.any(torch.isinf(new_momentum)):
#         print(f"[WARNING] round={current_round} class={class_id} | momentum 出錯，使用 prev")
#         new_momentum = prev_momentum.detach().clone()
#     elif mom_norm > max_mom_norm:
#         new_momentum = new_momentum * (max_mom_norm / (mom_norm + 1e-6))

#     global_model.logits_momentum[class_id] = new_momentum

#     avg_norm_ratio = min(1.0, adjusted_max_avg_norm / (avg_norm + 1e-6))
#     alpha_base = 1 / (1 + math.exp(-0.03 * (current_round - 60))) * 0.8
#     alpha = alpha_base * avg_norm_ratio
#     print(f"[DEBUG] round={current_round} class={class_id} | Pre-update avg: {avg.detach().cpu().numpy()}")
#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     dist_old_to_avg = (avg - old_wb).norm()
#     dist_old_to_updated = (updated - old_wb).norm()
#     ratio_thresh = 2.0 if current_round >= 100 else 1.5

#     if dist_old_to_updated > dist_old_to_avg * ratio_thresh:
#         print(f"[WARNING] round={current_round} class={class_id} | updated 偏離過大，使用 avg 取代")
#         updated = avg.detach().clone()

#     if updated.norm().item() < 1e-4:
#         print(f"[WARNING] round={current_round} class={class_id} | updated.norm 太小 fallback to avg")
#         updated = avg.detach().clone()

#     if not hasattr(global_model, "rolling_stats"):
#         global_model.rolling_stats = {
#             "acc_history": [],
#             "last_updated_class_params": {},
#             "last_valid_round": -1,
#             "panic_mode": False
#         }

#     global_model.rolling_stats["last_updated_class_params"][class_id] = old_wb.detach().clone()
#     global_model.rolling_stats["last_valid_round"] = current_round

#     if global_model.rolling_stats.get("panic_mode", False):
#         print(f"[PANIC] round={current_round} class={class_id} | 準確率崩潰，回復上次參數")
#         return global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())

#     print(f"[DEBUG] round={current_round:3d} | clients={values.size(0)} | ∆norm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | α={alpha:.2f} | avg.norm={avg_norm:.4f} | avg.mean={avg.mean().item():.4f}")
#     return updated


#------------------------------------
#可使用全部，但C8E4唯獨不行，會導致崩潰
# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
#     boost_interval: int = 70,
#     min_clients_threshold: int = 7,  # ✅ 保留不變
#     momentum_exit_round: int = 125,
#     local_epoch: int = 4
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device
#     num_clients = len(weights)

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)
#     avg = torch.sum(values * weights, dim=0)

#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([global_model.logits.weight[class_id], global_model.logits.bias[class_id].unsqueeze(0)])

#     # ✅ norm 安全界限處理
#     avg_norm_val = avg.norm()
#     norm_limit = 100.0
#     if avg_norm_val > norm_limit:
#         shrink_factor = norm_limit / (avg_norm_val + 1e-6)
#         avg = avg * shrink_factor + old_wb * (1 - shrink_factor)

#     if values.size(0) < min_clients_threshold and local_epoch >= 4:
#         if not hasattr(global_model, "history_avg"):
#             global_model.history_avg = {}
#         if class_id not in global_model.history_avg:
#             global_model.history_avg[class_id] = avg.detach().clone()
#         else:
#             prev_avg = global_model.history_avg[class_id]
#             diff = avg - prev_avg
#             diff_norm = diff.norm().item()
#             prev_norm = prev_avg.norm().item()
#             if prev_norm > 0 and diff_norm / prev_norm > 0.2:
#                 smoothing_factor = 0.5
#                 avg = prev_avg + smoothing_factor * diff
#             global_model.history_avg[class_id] = avg.detach().clone()

#     avg_norm = avg.norm().item()
#     if not hasattr(global_model, "history_norms"):
#         global_model.history_norms = []
#     global_model.history_norms.append(avg_norm)
#     if len(global_model.history_norms) > 15:
#         global_model.history_norms.pop(0)

#     min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
#     max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0
#     dynamic_range = max_avg_norm - min_avg_norm
#     adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

#     early_stage_momentum = (
#         values.size(0) < min_clients_threshold and
#         current_round < momentum_exit_round and
#         local_epoch >= 4
#     )

#     if values.size(0) < min_clients_threshold and not early_stage_momentum:
#         print(f"[INFO] round={current_round} class={class_id} | client 太少（{values.size(0)}），跳過 momentum")
#         return avg.detach().clone()

#     if avg_norm < 1e-3 or math.isnan(avg_norm) or math.isinf(avg_norm):
#         print(f"[WARNING] round={current_round} class={class_id} | avg 無效，fallback to old_wb")
#         return old_wb.detach().clone()

#     delta = avg - old_wb
#     if torch.any(torch.isnan(delta)) or torch.any(torch.isinf(delta)) or delta.norm() < 1e-6:
#         print(f"[WARNING] round={current_round} class={class_id} | delta 無效，使用 avg")
#         return avg.detach().clone()

#     if not hasattr(global_model, "history_avg_norms"):
#         global_model.history_avg_norms = {}
#     if not hasattr(global_model, "history_delta_norms"):
#         global_model.history_delta_norms = {}

#     global_model.history_avg_norms.setdefault(class_id, []).append(avg.norm().item())
#     global_model.history_delta_norms.setdefault(class_id, []).append(delta.norm().item())

#     if len(global_model.history_avg_norms[class_id]) > 10:
#         global_model.history_avg_norms[class_id].pop(0)
#     if len(global_model.history_delta_norms[class_id]) > 10:
#         global_model.history_delta_norms[class_id].pop(0)

#     def is_stable(seq: list, eps=1e-3):
#         return len(seq) >= 5 and all(abs(seq[i] - seq[i - 1]) < eps for i in range(1, len(seq)))

#     if is_stable(global_model.history_avg_norms[class_id]) and is_stable(global_model.history_delta_norms[class_id]):
#         noise = torch.randn_like(avg) * 0.01
#         avg = avg + noise
#         print(f"[NOISE] round={current_round} class={class_id} | learning stagnation detected, noise injected.")
#         delta = avg - old_wb

#     if current_round < 100:
#         scale_factor, delta_clip, base_beta, max_mom_norm = 0.3, 2.0, 0.9, 20.0
#     elif current_round < 200:
#         scale_factor, delta_clip, base_beta, max_mom_norm = 0.5, 4.0, 0.85, 30.0
#     else:
#         scale_factor, delta_clip, base_beta, max_mom_norm = 0.7, 6.0, 0.8, 35.0

#     client_scale_ratio = min(1.0, max(0.0, (num_clients - 2) / 18))
#     scale_factor = 0.1 + 0.6 * client_scale_ratio
#     base_beta = 0.95 - 0.15 * client_scale_ratio
#     delta_clip = 2.0 + 6.0 * client_scale_ratio
#     max_mom_norm = 10.0 + 40.0 * client_scale_ratio

#     # ✅ 防止 momentum_beta 太小
#     if early_stage_momentum:
#         momentum_beta = max(0.5, min(0.95, base_beta * (1.0 - current_round / (momentum_exit_round + 1e-6))))
#     else:
#         momentum_beta = min(0.95, base_beta)

#     # ✅ 避免 scale_factor 被 local_epoch 過度壓縮
#     local_epoch_scale = max(0.6, 1.0 / (1.0 + 0.2 * (local_epoch - 1)))
#     delta_clip *= local_epoch_scale
#     scale_factor *= local_epoch_scale

#     delta_norm_val = delta.norm()
#     if delta_norm_val > delta_clip:
#         delta = delta * (delta_clip / (delta_norm_val + 1e-6))
#     delta = delta * scale_factor

#     if not hasattr(global_model, "logits_momentum"):
#         global_model.logits_momentum = {}

#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta
#     mom_norm = new_momentum.norm()

#     if torch.any(torch.isnan(new_momentum)) or torch.any(torch.isinf(new_momentum)):
#         print(f"[WARNING] round={current_round} class={class_id} | momentum 出錯，使用 prev")
#         new_momentum = prev_momentum.detach().clone()
#     elif mom_norm > max_mom_norm:
#         print(f"[CLIP] round={current_round} class={class_id} | momentum norm ({mom_norm:.4f}) > {max_mom_norm}, 進行 clip")
#         new_momentum = new_momentum * (max_mom_norm / (mom_norm + 1e-6))

#     global_model.logits_momentum[class_id] = new_momentum

#     avg_norm_ratio = min(1.0, adjusted_max_avg_norm / (avg_norm + 1e-6))
#     alpha_base = 1 / (1 + math.exp(-0.03 * (current_round - 60))) * 0.8
#     alpha = alpha_base * avg_norm_ratio
#     print(f"[DEBUG] round={current_round} class={class_id} | Pre-update avg: {avg.detach().cpu().numpy()}")

#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     dist_old_to_avg = (avg - old_wb).norm()
#     dist_old_to_updated = (updated - old_wb).norm()
#     ratio_thresh = 2.5 if current_round >= 100 else 2.0
#     if dist_old_to_updated > dist_old_to_avg * ratio_thresh:
#         print(f"[WARNING] round={current_round} class={class_id} | updated 偏離過大，使用 avg")
#         updated = avg.detach().clone()

#     if updated.norm().item() < 1e-5:
#         print(f"[FALLBACK] round={current_round} class={class_id} | updated.norm 太小 fallback to avg")
#         updated = avg.detach().clone()

#     if not hasattr(global_model, "rolling_stats"):
#         global_model.rolling_stats = {
#             "acc_history": [],
#             "last_updated_class_params": {},
#             "last_valid_round": -1,
#             "panic_mode": False
#         }

#     global_model.rolling_stats["last_updated_class_params"][class_id] = old_wb.detach().clone()
#     global_model.rolling_stats["last_valid_round"] = current_round

#     if global_model.rolling_stats.get("panic_mode", False):
#         print(f"[PANIC] round={current_round} class={class_id} | 準確率崩潰，回復上次參數")
#         return global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())

#     print(f"[DEBUG] round={current_round:3d} | clients={values.size(0)} | ∆norm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | α={alpha:.2f} | avg.norm={avg_norm:.4f} | avg.mean={avg.mean().item():.4f}")
#     return updated


# import torch
# import math
# import numpy as np
# from typing import Union

# #--------------------------------
# #C32E4可用C16E4也可用C8E4會崩潰(413Round才崩潰)----->early clip版本
# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
#     boost_interval: int = 70,
#     min_clients_threshold: int = 7,
#     momentum_exit_round: int = 125,
#     local_epoch: int = 4
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device
#     num_clients = len(weights)

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)
#     avg = torch.sum(values * weights, dim=0)

#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([global_model.logits.weight[class_id], global_model.logits.bias[class_id].unsqueeze(0)])

#     avg_norm_val = avg.norm()
#     norm_limit = 100.0
#     if avg_norm_val > norm_limit:
#         shrink_factor = norm_limit / (avg_norm_val + 1e-6)
#         avg = avg * shrink_factor + old_wb * (1 - shrink_factor)

#     if values.size(0) < min_clients_threshold and local_epoch >= 4:
#         if not hasattr(global_model, "history_avg"):
#             global_model.history_avg = {}
#         if class_id not in global_model.history_avg:
#             global_model.history_avg[class_id] = avg.detach().clone()
#         else:
#             prev_avg = global_model.history_avg[class_id]
#             diff = avg - prev_avg
#             diff_norm = diff.norm().item()
#             prev_norm = prev_avg.norm().item()
#             if prev_norm > 0 and diff_norm / prev_norm > 0.2:
#                 smoothing_factor = 0.5
#                 avg = prev_avg + smoothing_factor * diff
#             global_model.history_avg[class_id] = avg.detach().clone()

#     avg_norm = avg.norm().item()
#     if not hasattr(global_model, "history_norms"):
#         global_model.history_norms = []
#     global_model.history_norms.append(avg_norm)
#     if len(global_model.history_norms) > 15:
#         global_model.history_norms.pop(0)

#     min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
#     max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0
#     dynamic_range = max_avg_norm - min_avg_norm
#     adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

#     # 🔍 提早偵測條件 1：delta 波動大 + client 太少 + local epoch 太多
#     if (
#         values.size(0) < min_clients_threshold and
#         local_epoch >= 4 and
#         delta.norm().item() > 3.0 * np.mean(global_model.history_delta_norms.get(class_id, [1.0]))
#     ):
#         print(f"[EARLY EXIT] round={current_round} class={class_id} | delta 太大且 client 太少，跳過聚合")
#         return old_wb.detach().clone()
        
#     early_stage_momentum = (
#         values.size(0) < min_clients_threshold and
#         current_round < momentum_exit_round and
#         local_epoch >= 4
#     )

#     if values.size(0) < min_clients_threshold and not early_stage_momentum:
#         print(f"[INFO] round={current_round} class={class_id} | client 太少（{values.size(0)}），跳過 momentum")
#         return avg.detach().clone()

#     if avg_norm < 1e-3 or math.isnan(avg_norm) or math.isinf(avg_norm):
#         print(f"[WARNING] round={current_round} class={class_id} | avg 無效，fallback to old_wb")
#         return old_wb.detach().clone()

#     delta = avg - old_wb
#     if torch.any(torch.isnan(delta)) or torch.any(torch.isinf(delta)) or delta.norm() < 1e-6:
#         print(f"[WARNING] round={current_round} class={class_id} | delta 無效，使用 avg")
#         return avg.detach().clone()

#     if not hasattr(global_model, "history_avg_norms"):
#         global_model.history_avg_norms = {}
#     if not hasattr(global_model, "history_delta_norms"):
#         global_model.history_delta_norms = {}

#     global_model.history_avg_norms.setdefault(class_id, []).append(avg.norm().item())
#     global_model.history_delta_norms.setdefault(class_id, []).append(delta.norm().item())
#     if len(global_model.history_avg_norms[class_id]) > 10:
#         global_model.history_avg_norms[class_id].pop(0)
#     if len(global_model.history_delta_norms[class_id]) > 10:
#         global_model.history_delta_norms[class_id].pop(0)

#     def is_stable(seq: list, eps=1e-3):
#         return len(seq) >= 5 and all(abs(seq[i] - seq[i - 1]) < eps for i in range(1, len(seq)))

#     if is_stable(global_model.history_avg_norms[class_id]) and is_stable(global_model.history_delta_norms[class_id]):
#         noise = torch.randn_like(avg) * 0.01
#         avg = avg + noise
#         print(f"[NOISE] round={current_round} class={class_id} | learning stagnation detected, noise injected.")
#         delta = avg - old_wb
#     # ✅ Early Stop 條件：早期輪次，客戶端太少或 epoch 太淺
#     if values.size(0) <= min_clients_threshold and local_epoch <= 4:
#         print(f"[EARLY STOP] round={current_round} class={class_id} | client={values.size(0)}, epoch={local_epoch} 條件過早，跳過")
#         return old_wb.detach().clone()

#     #此部分，常數應寫為公式，去使其參數可解釋
#     if current_round < 100:
#         scale_factor, delta_clip, base_beta, max_mom_norm = 0.3, 2.0, 0.9, 20.0
#     elif current_round < 200:
#         scale_factor, delta_clip, base_beta, max_mom_norm = 0.5, 4.0, 0.85, 30.0
#     else:
#         scale_factor, delta_clip, base_beta, max_mom_norm = 0.7, 6.0, 0.8, 35.0

#     client_scale_ratio = min(1.0, max(0.0, (num_clients - 2) / 18))
#     scale_factor = 0.1 + 0.6 * client_scale_ratio
#     base_beta = 0.95 - 0.15 * client_scale_ratio
#     delta_clip = 2.0 + 6.0 * client_scale_ratio
#     max_mom_norm = 10.0 + 40.0 * client_scale_ratio

#     if early_stage_momentum:
#         momentum_beta = max(0.5, min(0.95, base_beta * (1.0 - current_round / (momentum_exit_round + 1e-6))))
#     else:
#         momentum_beta = min(0.95, base_beta)

#     local_epoch_scale = max(0.6, 1.0 / (1.0 + 0.2 * (local_epoch - 1)))
#     delta_clip *= local_epoch_scale
#     scale_factor *= local_epoch_scale

#     # 🔧 新增 delta 穩定性偵測與自動降火機制
#     if not hasattr(global_model, "delta_stability"):
#         global_model.delta_stability = {}
#     global_model.delta_stability.setdefault(class_id, []).append(delta.norm().item())
#     if len(global_model.delta_stability[class_id]) > 5:
#         global_model.delta_stability[class_id].pop(0)

#     def delta_is_unstable(seq, threshold=1.5):
#         if len(seq) < 3:
#             return False
#         diffs = [abs(seq[i] - seq[i - 1]) for i in range(1, len(seq))]
#         return any(d > threshold * np.mean(seq) for d in diffs)

#     if delta_is_unstable(global_model.delta_stability[class_id]):
#         print(f"[STABILIZER] round={current_round} class={class_id} | Delta 波動過大，進行降火")
#         scale_factor *= 0.5
#         alpha = 0.3  # 明確降火 alpha
#         momentum_beta *= 0.8

#     delta_norm_val = delta.norm()
#     if delta_norm_val > delta_clip:
#         delta = delta * (delta_clip / (delta_norm_val + 1e-6))
#     delta = delta * scale_factor

#     if not hasattr(global_model, "logits_momentum"):
#         global_model.logits_momentum = {}
#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta
#     mom_norm = new_momentum.norm()

#     if torch.any(torch.isnan(new_momentum)) or torch.any(torch.isinf(new_momentum)):
#         print(f"[WARNING] round={current_round} class={class_id} | momentum 出錯，使用 prev")
#         new_momentum = prev_momentum.detach().clone()
#     elif mom_norm > max_mom_norm:
#         print(f"[CLIP] round={current_round} class={class_id} | momentum norm ({mom_norm:.4f}) > {max_mom_norm}, 進行 clip")
#         new_momentum = new_momentum * (max_mom_norm / (mom_norm + 1e-6))

#     global_model.logits_momentum[class_id] = new_momentum

#     avg_norm_ratio = min(1.0, adjusted_max_avg_norm / (avg_norm + 1e-6))
#     alpha_base = 1 / (1 + math.exp(-0.03 * (current_round - 60))) * 0.8
#     alpha = alpha_base * avg_norm_ratio
#     #--------------avg_norm若是太大，可做early stop跳過聚合
#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     # 🔧 round < 40 儲存 safe copy
#     if current_round < 40:
#         if not hasattr(global_model, "safe_copy"):
#             global_model.safe_copy = {}
#         global_model.safe_copy[class_id] = avg.detach().clone()

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     dist_old_to_avg = (avg - old_wb).norm()
#     dist_old_to_updated = (updated - old_wb).norm()
#     ratio_thresh = 2.5 if current_round >= 100 else 2.0
#     if dist_old_to_updated > dist_old_to_avg * ratio_thresh:
#         print(f"[WARNING] round={current_round} class={class_id} | updated 偏離過大，使用 avg")
#         updated = avg.detach().clone()

#     # 🔧 updated 崩潰 fallback
#     if updated.norm().item() < 1e-5 or torch.isnan(updated).any() or torch.isinf(updated).any() or updated.norm().item() > 300.0:
#         print(f"[FALLBACK] round={current_round} class={class_id} | updated 無效，回退 safe copy")
#         updated = global_model.safe_copy.get(class_id, avg.detach().clone())

#     if not hasattr(global_model, "rolling_stats"):
#         global_model.rolling_stats = {
#             "acc_history": [],
#             "last_updated_class_params": {},
#             "last_valid_round": -1,
#             "panic_mode": False
#         }

#     global_model.rolling_stats["last_updated_class_params"][class_id] = old_wb.detach().clone()
#     global_model.rolling_stats["last_valid_round"] = current_round

#     if global_model.rolling_stats.get("panic_mode", False):
#         print(f"[PANIC] round={current_round} class={class_id} | 準確率崩潰，回復上次參數")
#         return global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())

#     print(f"[DEBUG] round={current_round:3d} | clients={values.size(0)} | ∆norm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | α={alpha:.2f} | avg.norm={avg_norm:.4f} | avg.mean={avg.mean().item():.4f}")
    
#     # 🔍 Acc-based panic detection
#     if hasattr(global_model, "rolling_stats"):
#         acc_hist = global_model.rolling_stats.get("acc_history", [])
#         if len(acc_hist) >= 6:
#             recent_avg = np.mean(acc_hist[-3:])
#             prev_avg = np.mean(acc_hist[-6:-3])
#             if prev_avg > 0.3 and (recent_avg < 0.5 * prev_avg):
#                 print(f"[PANIC MODE] round={current_round} class={class_id} | Accuracy drop detected: {prev_avg:.4f} -> {recent_avg:.4f}")
#                 global_model.rolling_stats["panic_mode"] = True
#                 global_model.rolling_stats["panic_round"] = current_round
#                 global_model.rolling_stats["panic_class_id"] = class_id
#                 updated = global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())
#                 return updated

#     return updated

#--------------------------------
#針對C8E4 去做平滑，但是210Round就崩潰; C32、E4沒有問題
# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
#     boost_interval: int = 70,
#     min_clients_threshold: int = 7,
#     momentum_exit_round: int = 125,
#     local_epoch: int = 4
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device
#     num_clients = len(weights)

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)
#     avg = torch.sum(values * weights, dim=0)
    
#     # ===== [新增] avg 防爆處理 - value clipping =====
#     max_avg_val = 20.0  # 可調參數，預防過大激活值
#     avg = torch.clamp(avg, -max_avg_val, max_avg_val)

#     # ===== [新增] avg 防爆處理 - avg norm 二次檢查 =====
#     avg_norm_val = avg.norm()
#     if torch.isnan(avg_norm_val) or torch.isinf(avg_norm_val) or avg_norm_val > 200.0:
#         print(f"[FALLBACK] round={current_round} class={class_id} | avg.norm={avg_norm_val:.4f} 無效或爆炸，回退至 old_wb")
#         return old_wb.detach().clone()

#     # ===== [新增] avg 層級微縮（soft shrink）=====
#     shrink_thresh = 120.0
#     if avg.norm().item() > shrink_thresh:
#         shrink_factor = shrink_thresh / (avg.norm().item() + 1e-6)
#         avg = avg * shrink_factor + old_wb * (1 - shrink_factor)
#         print(f"[SHRINK] round={current_round} class={class_id} | avg.norm 過高進行 shrink factor={shrink_factor:.3f}")
#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([global_model.logits.weight[class_id], global_model.logits.bias[class_id].unsqueeze(0)])

#     avg_norm_val = avg.norm()
#     norm_limit = 100.0
#     if avg_norm_val > norm_limit:
#         shrink_factor = norm_limit / (avg_norm_val + 1e-6)
#         avg = avg * shrink_factor + old_wb * (1 - shrink_factor)

#     if values.size(0) < min_clients_threshold and local_epoch >= 4:
#         if not hasattr(global_model, "history_avg"):
#             global_model.history_avg = {}
#         if class_id not in global_model.history_avg:
#             global_model.history_avg[class_id] = avg.detach().clone()
#             prev_avg = avg.detach().clone()  # 🔧 fallback 定義
#         else:
#             prev_avg = global_model.history_avg[class_id]
#             diff = avg - prev_avg
#             diff_norm = diff.norm().item()
#             prev_norm = prev_avg.norm().item()
#             if prev_norm > 0 and diff_norm / prev_norm > 0.2:
#                 smoothing_factor = 0.5
#                 avg = prev_avg + smoothing_factor * diff
#             global_model.history_avg[class_id] = avg.detach().clone()
#     else:
#         prev_avg = avg.detach().clone()  # 🔧 fallback for downstream use


#     avg_norm = avg.norm().item()
#     if not hasattr(global_model, "history_norms"):
#         global_model.history_norms = []
#     global_model.history_norms.append(avg_norm)
#     if len(global_model.history_norms) > 15:
#         global_model.history_norms.pop(0)

#     min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
#     max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0
#     dynamic_range = max_avg_norm - min_avg_norm
#     adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

#     early_stage_momentum = (
#         values.size(0) < min_clients_threshold and
#         current_round < momentum_exit_round and
#         local_epoch >= 4
#     )

#     if values.size(0) < min_clients_threshold and not early_stage_momentum:
#         print(f"[INFO] round={current_round} class={class_id} | client 太少（{values.size(0)}），跳過 momentum")
#         return avg.detach().clone()
    

#     if avg_norm < 1e-3 or math.isnan(avg_norm) or math.isinf(avg_norm):
#         print(f"[WARNING] round={current_round} class={class_id} | avg 無效，fallback to old_wb")
#         return old_wb.detach().clone()

#     delta = avg - old_wb
#     if torch.any(torch.isnan(delta)) or torch.any(torch.isinf(delta)) or delta.norm() < 1e-6:
#         print(f"[WARNING] round={current_round} class={class_id} | delta 無效，使用 avg")
#         return avg.detach().clone()

#     if not hasattr(global_model, "history_avg_norms"):
#         global_model.history_avg_norms = {}
#     if not hasattr(global_model, "history_delta_norms"):
#         global_model.history_delta_norms = {}

#     global_model.history_avg_norms.setdefault(class_id, []).append(avg.norm().item())
#     global_model.history_delta_norms.setdefault(class_id, []).append(delta.norm().item())
#     if len(global_model.history_avg_norms[class_id]) > 10:
#         global_model.history_avg_norms[class_id].pop(0)
#     if len(global_model.history_delta_norms[class_id]) > 10:
#         global_model.history_delta_norms[class_id].pop(0)

#     def is_stable(seq: list, eps=1e-3):
#         return len(seq) >= 5 and all(abs(seq[i] - seq[i - 1]) < eps for i in range(1, len(seq)))

#     if (avg.norm().item() - prev_avg.norm().item()) > 10.0 and values.size(0) < min_clients_threshold:
#         print(f"[FALLBACK] round={current_round} class={class_id} | avg.norm 相對上一輪超出10，使用上一輪參數")
#         return old_wb.detach().clone()


#     if is_stable(global_model.history_avg_norms[class_id]) and is_stable(global_model.history_delta_norms[class_id]):
#         noise = torch.randn_like(avg) * 0.01
#         avg = avg + noise
#         print(f"[NOISE] round={current_round} class={class_id} | learning stagnation detected, noise injected.")
#         delta = avg - old_wb

#     if current_round < 100:
#         scale_factor, delta_clip, base_beta, max_mom_norm = 0.3, 2.0, 0.9, 20.0
#     elif current_round < 200:
#         scale_factor, delta_clip, base_beta, max_mom_norm = 0.5, 4.0, 0.85, 30.0
#     else:
#         scale_factor, delta_clip, base_beta, max_mom_norm = 0.7, 6.0, 0.8, 35.0

#     client_scale_ratio = min(1.0, max(0.0, (num_clients - 2) / 18))
#     scale_factor = 0.1 + 0.6 * client_scale_ratio
#     base_beta = 0.95 - 0.15 * client_scale_ratio
#     delta_clip = 2.0 + 6.0 * client_scale_ratio
#     max_mom_norm = 10.0 + 40.0 * client_scale_ratio

#     if early_stage_momentum:
#         momentum_beta = max(0.5, min(0.95, base_beta * (1.0 - current_round / (momentum_exit_round + 1e-6))))
#     else:
#         momentum_beta = min(0.95, base_beta)

#     local_epoch_scale = max(0.6, 1.0 / (1.0 + 0.2 * (local_epoch - 1)))
#     delta_clip *= local_epoch_scale
#     scale_factor *= local_epoch_scale

#     # 🔧 新增 delta 穩定性偵測與自動降火機制
#     if not hasattr(global_model, "delta_stability"):
#         global_model.delta_stability = {}
#     global_model.delta_stability.setdefault(class_id, []).append(delta.norm().item())
#     if len(global_model.delta_stability[class_id]) > 5:
#         global_model.delta_stability[class_id].pop(0)

#     def delta_is_unstable(seq, threshold=1.5):
#         if len(seq) < 3:
#             return False
#         diffs = [abs(seq[i] - seq[i - 1]) for i in range(1, len(seq))]
#         return any(d > threshold * np.mean(seq) for d in diffs)

#     if delta_is_unstable(global_model.delta_stability[class_id]):
#         print(f"[STABILIZER] round={current_round} class={class_id} | Delta 波動過大，進行降火")
#         scale_factor *= 0.5
#         alpha = 0.3  # 明確降火 alpha
#         momentum_beta *= 0.8

#     delta_norm_val = delta.norm()
#     if delta_norm_val > delta_clip:
#         delta = delta * (delta_clip / (delta_norm_val + 1e-6))
#     delta = delta * scale_factor

#     if not hasattr(global_model, "logits_momentum"):
#         global_model.logits_momentum = {}
#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta
#     mom_norm = new_momentum.norm()

#     if torch.any(torch.isnan(new_momentum)) or torch.any(torch.isinf(new_momentum)):
#         print(f"[WARNING] round={current_round} class={class_id} | momentum 出錯，使用 prev")
#         new_momentum = prev_momentum.detach().clone()
#     elif mom_norm > max_mom_norm:
#         print(f"[CLIP] round={current_round} class={class_id} | momentum norm ({mom_norm:.4f}) > {max_mom_norm}, 進行 clip")
#         new_momentum = new_momentum * (max_mom_norm / (mom_norm + 1e-6))

#     global_model.logits_momentum[class_id] = new_momentum

#     avg_norm_ratio = min(1.0, adjusted_max_avg_norm / (avg_norm + 1e-6))
#     alpha_base = 1 / (1 + math.exp(-0.03 * (current_round - 60))) * 0.8
#     alpha = alpha_base * avg_norm_ratio

#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     # 🔧 round < 40 儲存 safe copy
#     if current_round < 40:
#         if not hasattr(global_model, "safe_copy"):
#             global_model.safe_copy = {}
#         global_model.safe_copy[class_id] = avg.detach().clone()

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     dist_old_to_avg = (avg - old_wb).norm()
#     dist_old_to_updated = (updated - old_wb).norm()
#     ratio_thresh = 2.5 if current_round >= 100 else 2.0
#     if dist_old_to_updated > dist_old_to_avg * ratio_thresh:
#         print(f"[WARNING] round={current_round} class={class_id} | updated 偏離過大，使用 avg")
#         updated = avg.detach().clone()

#     # 🔧 updated 崩潰 fallback
#     if updated.norm().item() < 1e-5 or torch.isnan(updated).any() or torch.isinf(updated).any() or updated.norm().item() > 300.0:
#         print(f"[FALLBACK] round={current_round} class={class_id} | updated 無效，回退 safe copy")
#         updated = global_model.safe_copy.get(class_id, avg.detach().clone())

#     if not hasattr(global_model, "rolling_stats"):
#         global_model.rolling_stats = {
#             "acc_history": [],
#             "last_updated_class_params": {},
#             "last_valid_round": -1,
#             "panic_mode": False
#         }

#     global_model.rolling_stats["last_updated_class_params"][class_id] = old_wb.detach().clone()
#     global_model.rolling_stats["last_valid_round"] = current_round

#     if global_model.rolling_stats.get("panic_mode", False):
#         print(f"[PANIC] round={current_round} class={class_id} | 準確率崩潰，回復上次參數")
#         return global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())

#     print(f"[DEBUG] round={current_round:3d} | clients={values.size(0)} | ∆norm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | α={alpha:.2f} | avg.norm={avg_norm:.4f} | avg.mean={avg.mean().item():.4f}")
#     return updated

#---------------------------
#加強版防爆AVG-->for C8E4不要崩潰
# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
#     boost_interval: int = 70,
#     min_clients_threshold: int = 7,
#     momentum_exit_round: int = 125,
#     local_epoch: int = 4
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device
#     num_clients = len(weights)

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)
#     avg = torch.sum(values * weights, dim=0)
    
#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([global_model.logits.weight[class_id], global_model.logits.bias[class_id].unsqueeze(0)])

#     if values.size(0) < min_clients_threshold and local_epoch >= 4:
#         # ===== [新增] avg 防爆處理 - value clipping =====
#         max_avg_val = 20.0  # 可調參數，預防過大激活值
#         avg = torch.clamp(avg, -max_avg_val, max_avg_val)

#         # ===== [新增] avg 防爆處理 - avg norm 二次檢查 =====
#         avg_norm_val = avg.norm()
#         if torch.isnan(avg_norm_val) or torch.isinf(avg_norm_val) or avg_norm_val > 200.0:
#             print(f"[FALLBACK] round={current_round} class={class_id} | avg.norm={avg_norm_val:.4f} 無效或爆炸，回退至 old_wb")
#             return old_wb.detach().clone()

#         # ===== [新增] avg 層級微縮（soft shrink）=====
#         shrink_thresh = 120.0
#         if avg.norm().item() > shrink_thresh:
#             shrink_factor = shrink_thresh / (avg.norm().item() + 1e-6)
#             avg = avg * shrink_factor + old_wb * (1 - shrink_factor)
#             print(f"[SHRINK] round={current_round} class={class_id} | avg.norm 過高進行 shrink factor={shrink_factor:.3f}")
    
    

#     avg_norm_val = avg.norm()
#     norm_limit = 100.0
#     if avg_norm_val > norm_limit:
#         shrink_factor = norm_limit / (avg_norm_val + 1e-6)
#         avg = avg * shrink_factor + old_wb * (1 - shrink_factor)

#     if values.size(0) < min_clients_threshold and local_epoch >= 4:
#         if not hasattr(global_model, "history_avg"):
#             global_model.history_avg = {}
#         if class_id not in global_model.history_avg:
#             global_model.history_avg[class_id] = avg.detach().clone()
#             prev_avg = avg.detach().clone()  # 🔧 fallback 定義
#         else:
#             prev_avg = global_model.history_avg[class_id]
#             diff = avg - prev_avg
#             diff_norm = diff.norm().item()
#             prev_norm = prev_avg.norm().item()
#             if prev_norm > 0 and diff_norm / prev_norm > 0.2:
#                 smoothing_factor = 0.5
#                 avg = prev_avg + smoothing_factor * diff
#             global_model.history_avg[class_id] = avg.detach().clone()
#     else:
#         prev_avg = avg.detach().clone()  # 🔧 fallback for downstream use


#     avg_norm = avg.norm().item()
#     if not hasattr(global_model, "history_norms"):
#         global_model.history_norms = []
#     global_model.history_norms.append(avg_norm)
#     if len(global_model.history_norms) > 15:
#         global_model.history_norms.pop(0)

#     min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
#     max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0
#     dynamic_range = max_avg_norm - min_avg_norm
#     adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

#     early_stage_momentum = (
#         values.size(0) < min_clients_threshold and
#         current_round < momentum_exit_round and
#         local_epoch >= 4
#     )

#     if values.size(0) < min_clients_threshold and not early_stage_momentum:
#         print(f"[INFO] round={current_round} class={class_id} | client 太少（{values.size(0)}），跳過 momentum")
#         return avg.detach().clone()
    

#     if avg_norm < 1e-3 or math.isnan(avg_norm) or math.isinf(avg_norm):
#         print(f"[WARNING] round={current_round} class={class_id} | avg 無效，fallback to old_wb")
#         return old_wb.detach().clone()

#     delta = avg - old_wb
#     if torch.any(torch.isnan(delta)) or torch.any(torch.isinf(delta)) or delta.norm() < 1e-6:
#         print(f"[WARNING] round={current_round} class={class_id} | delta 無效，使用 avg")
#         return avg.detach().clone()

#     if not hasattr(global_model, "history_avg_norms"):
#         global_model.history_avg_norms = {}
#     if not hasattr(global_model, "history_delta_norms"):
#         global_model.history_delta_norms = {}

#     global_model.history_avg_norms.setdefault(class_id, []).append(avg.norm().item())
#     global_model.history_delta_norms.setdefault(class_id, []).append(delta.norm().item())
#     if len(global_model.history_avg_norms[class_id]) > 10:
#         global_model.history_avg_norms[class_id].pop(0)
#     if len(global_model.history_delta_norms[class_id]) > 10:
#         global_model.history_delta_norms[class_id].pop(0)

#     def is_stable(seq: list, eps=1e-3):
#         return len(seq) >= 5 and all(abs(seq[i] - seq[i - 1]) < eps for i in range(1, len(seq)))

#     if (avg.norm().item() - prev_avg.norm().item()) > 10.0 and values.size(0) < min_clients_threshold:
#         print(f"[FALLBACK] round={current_round} class={class_id} | avg.norm 相對上一輪超出10，使用上一輪參數")
#         return old_wb.detach().clone()


#     if is_stable(global_model.history_avg_norms[class_id]) and is_stable(global_model.history_delta_norms[class_id]):
#         noise = torch.randn_like(avg) * 0.01
#         avg = avg + noise
#         print(f"[NOISE] round={current_round} class={class_id} | learning stagnation detected, noise injected.")
#         delta = avg - old_wb

#     if current_round < 100:
#         scale_factor, delta_clip, base_beta, max_mom_norm = 0.3, 2.0, 0.9, 20.0
#     elif current_round < 200:
#         scale_factor, delta_clip, base_beta, max_mom_norm = 0.5, 4.0, 0.85, 30.0
#     else:
#         scale_factor, delta_clip, base_beta, max_mom_norm = 0.7, 6.0, 0.8, 35.0

#     client_scale_ratio = min(1.0, max(0.0, (num_clients - 2) / 18))
#     scale_factor = 0.1 + 0.6 * client_scale_ratio
#     base_beta = 0.95 - 0.15 * client_scale_ratio
#     delta_clip = 2.0 + 6.0 * client_scale_ratio
#     max_mom_norm = 10.0 + 40.0 * client_scale_ratio

#     if early_stage_momentum:
#         momentum_beta = max(0.5, min(0.95, base_beta * (1.0 - current_round / (momentum_exit_round + 1e-6))))
#     else:
#         momentum_beta = min(0.95, base_beta)

#     local_epoch_scale = max(0.6, 1.0 / (1.0 + 0.2 * (local_epoch - 1)))
#     delta_clip *= local_epoch_scale
#     scale_factor *= local_epoch_scale

#     # 🔧 新增 delta 穩定性偵測與自動降火機制
#     if not hasattr(global_model, "delta_stability"):
#         global_model.delta_stability = {}
#     global_model.delta_stability.setdefault(class_id, []).append(delta.norm().item())
#     if len(global_model.delta_stability[class_id]) > 5:
#         global_model.delta_stability[class_id].pop(0)

#     def delta_is_unstable(seq, threshold=1.5):
#         if len(seq) < 3:
#             return False
#         diffs = [abs(seq[i] - seq[i - 1]) for i in range(1, len(seq))]
#         return any(d > threshold * np.mean(seq) for d in diffs)

#     if delta_is_unstable(global_model.delta_stability[class_id]):
#         print(f"[STABILIZER] round={current_round} class={class_id} | Delta 波動過大，進行降火")
#         scale_factor *= 0.5
#         alpha = 0.3  # 明確降火 alpha
#         momentum_beta *= 0.8

#     delta_norm_val = delta.norm()
#     if delta_norm_val > delta_clip:
#         delta = delta * (delta_clip / (delta_norm_val + 1e-6))
#     delta = delta * scale_factor

#     if not hasattr(global_model, "logits_momentum"):
#         global_model.logits_momentum = {}
#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta
#     mom_norm = new_momentum.norm()

#     if torch.any(torch.isnan(new_momentum)) or torch.any(torch.isinf(new_momentum)):
#         print(f"[WARNING] round={current_round} class={class_id} | momentum 出錯，使用 prev")
#         new_momentum = prev_momentum.detach().clone()
#     elif mom_norm > max_mom_norm:
#         print(f"[CLIP] round={current_round} class={class_id} | momentum norm ({mom_norm:.4f}) > {max_mom_norm}, 進行 clip")
#         new_momentum = new_momentum * (max_mom_norm / (mom_norm + 1e-6))

#     global_model.logits_momentum[class_id] = new_momentum

#     avg_norm_ratio = min(1.0, adjusted_max_avg_norm / (avg_norm + 1e-6))
#     alpha_base = 1 / (1 + math.exp(-0.03 * (current_round - 60))) * 0.8
#     alpha = alpha_base * avg_norm_ratio

#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     # 🔧 round < 40 儲存 safe copy
#     if current_round < 40:
#         if not hasattr(global_model, "safe_copy"):
#             global_model.safe_copy = {}
#         global_model.safe_copy[class_id] = avg.detach().clone()

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     dist_old_to_avg = (avg - old_wb).norm()
#     dist_old_to_updated = (updated - old_wb).norm()
#     ratio_thresh = 2.5 if current_round >= 100 else 2.0
#     if dist_old_to_updated > dist_old_to_avg * ratio_thresh:
#         print(f"[WARNING] round={current_round} class={class_id} | updated 偏離過大，使用 avg")
#         updated = avg.detach().clone()

#     # 🔧 updated 崩潰 fallback
#     if updated.norm().item() < 1e-5 or torch.isnan(updated).any() or torch.isinf(updated).any() or updated.norm().item() > 300.0:
#         print(f"[FALLBACK] round={current_round} class={class_id} | updated 無效，回退 safe copy")
#         updated = global_model.safe_copy.get(class_id, avg.detach().clone())

#     if not hasattr(global_model, "rolling_stats"):
#         global_model.rolling_stats = {
#             "acc_history": [],
#             "last_updated_class_params": {},
#             "last_valid_round": -1,
#             "panic_mode": False
#         }

#     global_model.rolling_stats["last_updated_class_params"][class_id] = old_wb.detach().clone()
#     global_model.rolling_stats["last_valid_round"] = current_round

#     if global_model.rolling_stats.get("panic_mode", False):
#         print(f"[PANIC] round={current_round} class={class_id} | 準確率崩潰，回復上次參數")
#         return global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())

#     print(f"[DEBUG] round={current_round:3d} | clients={values.size(0)} | ∆norm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | α={alpha:.2f} | avg.norm={avg_norm:.4f} | avg.mean={avg.mean().item():.4f}")
#     return updated

#-----------------------
#修正條件使100Round可以往上提升
# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
#     boost_interval: int = 70,
#     min_clients_threshold: int = 7,
#     momentum_exit_round: int = 125,
#     local_epoch: int = 4
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device
#     num_clients = len(weights)

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)
#     avg = torch.sum(values * weights, dim=0)
    
#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([global_model.logits.weight[class_id], global_model.logits.bias[class_id].unsqueeze(0)])

#     if values.size(0) < min_clients_threshold and local_epoch >= 4:
#         # ===== [新增] avg 防爆處理 - value clipping =====
#         max_avg_val = 20.0  # 可調參數，預防過大激活值
#         avg = torch.clamp(avg, -max_avg_val, max_avg_val)

#         # ===== [新增] avg 防爆處理 - avg norm 二次檢查 =====
#         avg_norm_val = avg.norm()
#         if torch.isnan(avg_norm_val) or torch.isinf(avg_norm_val) or avg_norm_val > 200.0:
#             print(f"[FALLBACK] round={current_round} class={class_id} | avg.norm={avg_norm_val:.4f} 無效或爆炸，回退至 old_wb")
#             return old_wb.detach().clone()

#         # ===== [新增] avg 層級微縮（soft shrink）=====
#         shrink_thresh = 120.0
#         if avg.norm().item() > shrink_thresh:
#             shrink_factor = shrink_thresh / (avg.norm().item() + 1e-6)
#             avg = avg * shrink_factor + old_wb * (1 - shrink_factor)
#             print(f"[SHRINK] round={current_round} class={class_id} | avg.norm 過高進行 shrink factor={shrink_factor:.3f}")
    
    

#     avg_norm_val = avg.norm()
#     norm_limit = 100.0
#     if avg_norm_val > norm_limit:
#         shrink_factor = norm_limit / (avg_norm_val + 1e-6)
#         avg = avg * shrink_factor + old_wb * (1 - shrink_factor)

#     if values.size(0) < min_clients_threshold and local_epoch >= 4:
#         if not hasattr(global_model, "history_avg"):
#             global_model.history_avg = {}
#         if class_id not in global_model.history_avg:
#             global_model.history_avg[class_id] = avg.detach().clone()
#             prev_avg = avg.detach().clone()  # 🔧 fallback 定義
#         else:
#             prev_avg = global_model.history_avg[class_id]
#             diff = avg - prev_avg
#             diff_norm = diff.norm().item()
#             prev_norm = prev_avg.norm().item()
#             if prev_norm > 0 and diff_norm / prev_norm > 0.2:
#                 smoothing_factor = 0.5
#                 avg = prev_avg + smoothing_factor * diff
#             global_model.history_avg[class_id] = avg.detach().clone()
#     else:
#         prev_avg = avg.detach().clone()  # 🔧 fallback for downstream use


#     avg_norm = avg.norm().item()
#     if not hasattr(global_model, "history_norms"):
#         global_model.history_norms = []
#     global_model.history_norms.append(avg_norm)
#     if len(global_model.history_norms) > 15:
#         global_model.history_norms.pop(0)

#     min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
#     max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0
#     dynamic_range = max_avg_norm - min_avg_norm
#     adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

#     early_stage_momentum = (
#         values.size(0) < min_clients_threshold and
#         current_round < momentum_exit_round and
#         local_epoch >= 4
#     )

#     if values.size(0) < min_clients_threshold and not early_stage_momentum:
#         print(f"[INFO] round={current_round} class={class_id} | client 太少（{values.size(0)}），跳過 momentum")
#         return avg.detach().clone()
    

#     if avg_norm < 1e-3 or math.isnan(avg_norm) or math.isinf(avg_norm):
#         print(f"[WARNING] round={current_round} class={class_id} | avg 無效，fallback to old_wb")
#         return old_wb.detach().clone()

#     delta = avg - old_wb
#     if torch.any(torch.isnan(delta)) or torch.any(torch.isinf(delta)) or delta.norm() < 1e-6:
#         print(f"[WARNING] round={current_round} class={class_id} | delta 無效，使用 avg")
#         return avg.detach().clone()

#     if not hasattr(global_model, "history_avg_norms"):
#         global_model.history_avg_norms = {}
#     if not hasattr(global_model, "history_delta_norms"):
#         global_model.history_delta_norms = {}

#     global_model.history_avg_norms.setdefault(class_id, []).append(avg.norm().item())
#     global_model.history_delta_norms.setdefault(class_id, []).append(delta.norm().item())
#     if len(global_model.history_avg_norms[class_id]) > 10:
#         global_model.history_avg_norms[class_id].pop(0)
#     if len(global_model.history_delta_norms[class_id]) > 10:
#         global_model.history_delta_norms[class_id].pop(0)

#     def is_stable(seq: list, eps=1e-3):
#         return len(seq) >= 5 and all(abs(seq[i] - seq[i - 1]) < eps for i in range(1, len(seq)))

#     if (avg.norm().item() - prev_avg.norm().item()) > 10.0 and values.size(0) < min_clients_threshold:
#         print(f"[FALLBACK] round={current_round} class={class_id} | avg.norm 相對上一輪超出10，使用上一輪參數")
#         return old_wb.detach().clone()


#     if is_stable(global_model.history_avg_norms[class_id]) and is_stable(global_model.history_delta_norms[class_id]):
#         noise = torch.randn_like(avg) * 0.01
#         avg = avg + noise
#         print(f"[NOISE] round={current_round} class={class_id} | learning stagnation detected, noise injected.")
#         delta = avg - old_wb

#     if current_round < 100:
#         scale_factor, delta_clip, base_beta, max_mom_norm = 0.3, 2.0, 0.9, 20.0
#     elif current_round < 200:
#         scale_factor, delta_clip, base_beta, max_mom_norm = 0.5, 4.0, 0.85, 30.0
#     else:
#         scale_factor, delta_clip, base_beta, max_mom_norm = 0.7, 6.0, 0.8, 35.0

#     client_scale_ratio = min(1.0, max(0.0, (num_clients - 2) / 18))
#     scale_factor = 0.1 + 0.6 * client_scale_ratio
#     base_beta = 0.95 - 0.15 * client_scale_ratio
#     delta_clip = 2.0 + 6.0 * client_scale_ratio
#     max_mom_norm = 10.0 + 40.0 * client_scale_ratio

#     if early_stage_momentum:
#         momentum_beta = max(0.5, min(0.95, base_beta * (1.0 - current_round / (momentum_exit_round + 1e-6))))
#     else:
#         momentum_beta = min(0.95, base_beta)

#     local_epoch_scale = max(0.6, 1.0 / (1.0 + 0.2 * (local_epoch - 1)))
#     delta_clip *= local_epoch_scale
#     scale_factor *= local_epoch_scale

#     # 🔧 新增 delta 穩定性偵測與自動降火機制
#     if not hasattr(global_model, "delta_stability"):
#         global_model.delta_stability = {}
#     global_model.delta_stability.setdefault(class_id, []).append(delta.norm().item())
#     if len(global_model.delta_stability[class_id]) > 5:
#         global_model.delta_stability[class_id].pop(0)

#     def delta_is_unstable(seq, threshold=1.5):
#         if len(seq) < 3:
#             return False
#         diffs = [abs(seq[i] - seq[i - 1]) for i in range(1, len(seq))]
#         return any(d > threshold * np.mean(seq) for d in diffs)

#     if delta_is_unstable(global_model.delta_stability[class_id]):
#         print(f"[STABILIZER] round={current_round} class={class_id} | Delta 波動過大，進行降火")
#         scale_factor *= 0.5
#         alpha = 0.3  # 明確降火 alpha
#         momentum_beta *= 0.8

#     delta_norm_val = delta.norm()
#     if delta_norm_val > delta_clip:
#         delta = delta * (delta_clip / (delta_norm_val + 1e-6))
#     delta = delta * scale_factor

#     if not hasattr(global_model, "logits_momentum"):
#         global_model.logits_momentum = {}
#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta
#     mom_norm = new_momentum.norm()

#     if torch.any(torch.isnan(new_momentum)) or torch.any(torch.isinf(new_momentum)):
#         print(f"[WARNING] round={current_round} class={class_id} | momentum 出錯，使用 prev")
#         new_momentum = prev_momentum.detach().clone()
#     elif mom_norm > max_mom_norm:
#         print(f"[CLIP] round={current_round} class={class_id} | momentum norm ({mom_norm:.4f}) > {max_mom_norm}, 進行 clip")
#         new_momentum = new_momentum * (max_mom_norm / (mom_norm + 1e-6))

#     global_model.logits_momentum[class_id] = new_momentum

#     avg_norm_ratio = min(1.0, adjusted_max_avg_norm / (avg_norm + 1e-6))
#     alpha_base = 1 / (1 + math.exp(-0.03 * (current_round - 60))) * 0.8
#     alpha = alpha_base * avg_norm_ratio

#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     # 🔧 round < 40 儲存 safe copy
#     if current_round < 40:
#         if not hasattr(global_model, "safe_copy"):
#             global_model.safe_copy = {}
#         global_model.safe_copy[class_id] = avg.detach().clone()

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     dist_old_to_avg = (avg - old_wb).norm()
#     dist_old_to_updated = (updated - old_wb).norm()
#     ratio_thresh = 2.5 if current_round >= 100 else 2.0
#     if dist_old_to_updated > dist_old_to_avg * ratio_thresh:
#         print(f"[WARNING] round={current_round} class={class_id} | updated 偏離過大，使用 avg")
#         updated = avg.detach().clone()

#     # 🔧 updated 崩潰 fallback
#     if updated.norm().item() < 1e-5 or torch.isnan(updated).any() or torch.isinf(updated).any() or updated.norm().item() > 300.0:
#         print(f"[FALLBACK] round={current_round} class={class_id} | updated 無效，回退 safe copy")
#         updated = global_model.safe_copy.get(class_id, avg.detach().clone())

#     if not hasattr(global_model, "rolling_stats"):
#         global_model.rolling_stats = {
#             "acc_history": [],
#             "last_updated_class_params": {},
#             "last_valid_round": -1,
#             "panic_mode": False
#         }

#     global_model.rolling_stats["last_updated_class_params"][class_id] = old_wb.detach().clone()
#     global_model.rolling_stats["last_valid_round"] = current_round

#     if global_model.rolling_stats.get("panic_mode", False):
#         print(f"[PANIC] round={current_round} class={class_id} | 準確率崩潰，回復上次參數")
#         return global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())

#     print(f"[DEBUG] round={current_round:3d} | clients={values.size(0)} | ∆norm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | α={alpha:.2f} | avg.norm={avg_norm:.4f} | avg.mean={avg.mean().item():.4f}")
#     return updated


#early clip修改版。--->成效不太好
# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
#     boost_interval: int = 70,
#     min_clients_threshold: int = 7,
#     momentum_exit_round: int = 125,
#     local_epoch: int = 4
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device
#     num_clients = len(weights)

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)
#     avg = torch.sum(values * weights, dim=0)

#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([global_model.logits.weight[class_id], global_model.logits.bias[class_id].unsqueeze(0)])

#     avg_norm_val = avg.norm()
#     norm_limit = 100.0
#     if avg_norm_val > norm_limit:
#         shrink_factor = norm_limit / (avg_norm_val + 1e-6)
#         avg = avg * shrink_factor + old_wb * (1 - shrink_factor)

#     if values.size(0) < min_clients_threshold and local_epoch >= 4:
#         if not hasattr(global_model, "history_avg"):
#             global_model.history_avg = {}
#         if class_id not in global_model.history_avg:
#             global_model.history_avg[class_id] = avg.detach().clone()
#         else:
#             prev_avg = global_model.history_avg[class_id]
#             diff = avg - prev_avg
#             diff_norm = diff.norm().item()
#             prev_norm = prev_avg.norm().item()
#             if prev_norm > 0 and diff_norm / prev_norm > 0.2:
#                 smoothing_factor = 0.5
#                 avg = prev_avg + smoothing_factor * diff
#             global_model.history_avg[class_id] = avg.detach().clone()

#     avg_norm = avg.norm().item()
#     if not hasattr(global_model, "history_norms"):
#         global_model.history_norms = []
#     global_model.history_norms.append(avg_norm)
#     if len(global_model.history_norms) > 15:
#         global_model.history_norms.pop(0)

#     min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
#     max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0
#     dynamic_range = max_avg_norm - min_avg_norm
#     adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

#     # 🔍 提早偵測條件 1：delta 波動大 + client 太少 + local epoch 太多
#     if (
#         values.size(0) < min_clients_threshold and
#         local_epoch >= 4 and
#         delta.norm().item() > 3.0 * np.mean(global_model.history_delta_norms.get(class_id, [1.0]))
#     ):
#         print(f"[EARLY EXIT] round={current_round} class={class_id} | delta 太大且 client 太少，跳過聚合")
#         return old_wb.detach().clone()
        
#     early_stage_momentum = (
#         values.size(0) < min_clients_threshold and
#         current_round < momentum_exit_round and
#         local_epoch >= 4
#     )

#     if values.size(0) < min_clients_threshold and not early_stage_momentum:
#         print(f"[INFO] round={current_round} class={class_id} | client 太少（{values.size(0)}），跳過 momentum")
#         return avg.detach().clone()

#     if avg_norm < 1e-3 or math.isnan(avg_norm) or math.isinf(avg_norm):
#         print(f"[WARNING] round={current_round} class={class_id} | avg 無效，fallback to old_wb")
#         return old_wb.detach().clone()

#     delta = avg - old_wb
#     if torch.any(torch.isnan(delta)) or torch.any(torch.isinf(delta)) or delta.norm() < 1e-6:
#         print(f"[WARNING] round={current_round} class={class_id} | delta 無效，使用 avg")
#         return avg.detach().clone()

#     if not hasattr(global_model, "history_avg_norms"):
#         global_model.history_avg_norms = {}
#     if not hasattr(global_model, "history_delta_norms"):
#         global_model.history_delta_norms = {}

#     global_model.history_avg_norms.setdefault(class_id, []).append(avg.norm().item())
#     global_model.history_delta_norms.setdefault(class_id, []).append(delta.norm().item())
#     if len(global_model.history_avg_norms[class_id]) > 10:
#         global_model.history_avg_norms[class_id].pop(0)
#     if len(global_model.history_delta_norms[class_id]) > 10:
#         global_model.history_delta_norms[class_id].pop(0)

#     def is_stable(seq: list, eps=1e-3):
#         return len(seq) >= 5 and all(abs(seq[i] - seq[i - 1]) < eps for i in range(1, len(seq)))

#     if is_stable(global_model.history_avg_norms[class_id]) and is_stable(global_model.history_delta_norms[class_id]):
#         noise = torch.randn_like(avg) * 0.01
#         avg = avg + noise
#         print(f"[NOISE] round={current_round} class={class_id} | learning stagnation detected, noise injected.")
#         delta = avg - old_wb
#     # ✅ Early Stop 條件：早期輪次，客戶端太少或 epoch 太淺
#     if values.size(0) <= min_clients_threshold and local_epoch <= 4:
#         print(f"[EARLY STOP] round={current_round} class={class_id} | client={values.size(0)}, epoch={local_epoch} 條件過早，跳過")
#         return old_wb.detach().clone()

#     #此部分，常數應寫為公式，去使其參數可解釋
#     if current_round < 100:
#         scale_factor, delta_clip, base_beta, max_mom_norm = 0.3, 2.0, 0.9, 20.0
#     elif current_round < 200:
#         scale_factor, delta_clip, base_beta, max_mom_norm = 0.5, 4.0, 0.85, 30.0
#     else:
#         scale_factor, delta_clip, base_beta, max_mom_norm = 0.7, 6.0, 0.8, 35.0

#     client_scale_ratio = min(1.0, max(0.0, (num_clients - 2) / 18))
#     scale_factor = 0.1 + 0.6 * client_scale_ratio
#     base_beta = 0.95 - 0.15 * client_scale_ratio
#     delta_clip = 2.0 + 6.0 * client_scale_ratio
#     max_mom_norm = 10.0 + 40.0 * client_scale_ratio

#     if early_stage_momentum:
#         momentum_beta = max(0.5, min(0.95, base_beta * (1.0 - current_round / (momentum_exit_round + 1e-6))))
#     else:
#         momentum_beta = min(0.95, base_beta)

#     local_epoch_scale = max(0.6, 1.0 / (1.0 + 0.2 * (local_epoch - 1)))
#     delta_clip *= local_epoch_scale
#     scale_factor *= local_epoch_scale

#     # 🔧 新增 delta 穩定性偵測與自動降火機制
#     if not hasattr(global_model, "delta_stability"):
#         global_model.delta_stability = {}
#     global_model.delta_stability.setdefault(class_id, []).append(delta.norm().item())
#     if len(global_model.delta_stability[class_id]) > 5:
#         global_model.delta_stability[class_id].pop(0)

#     def delta_is_unstable(seq, threshold=1.5):
#         if len(seq) < 3:
#             return False
#         diffs = [abs(seq[i] - seq[i - 1]) for i in range(1, len(seq))]
#         return any(d > threshold * np.mean(seq) for d in diffs)

#     if delta_is_unstable(global_model.delta_stability[class_id]):
#         print(f"[STABILIZER] round={current_round} class={class_id} | Delta 波動過大，進行降火")
#         scale_factor *= 0.5
#         alpha = 0.3  # 明確降火 alpha
#         momentum_beta *= 0.8

#     delta_norm_val = delta.norm()
#     if delta_norm_val > delta_clip:
#         delta = delta * (delta_clip / (delta_norm_val + 1e-6))
#     delta = delta * scale_factor

#     if not hasattr(global_model, "logits_momentum"):
#         global_model.logits_momentum = {}
#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta
#     mom_norm = new_momentum.norm()

#     if torch.any(torch.isnan(new_momentum)) or torch.any(torch.isinf(new_momentum)):
#         print(f"[WARNING] round={current_round} class={class_id} | momentum 出錯，使用 prev")
#         new_momentum = prev_momentum.detach().clone()
#     elif mom_norm > max_mom_norm:
#         print(f"[CLIP] round={current_round} class={class_id} | momentum norm ({mom_norm:.4f}) > {max_mom_norm}, 進行 clip")
#         new_momentum = new_momentum * (max_mom_norm / (mom_norm + 1e-6))

#     global_model.logits_momentum[class_id] = new_momentum

#     avg_norm_ratio = min(1.0, adjusted_max_avg_norm / (avg_norm + 1e-6))
#     alpha_base = 1 / (1 + math.exp(-0.03 * (current_round - 60))) * 0.8
#     alpha = alpha_base * avg_norm_ratio
#     #--------------avg_norm若是太大，可做early stop跳過聚合
#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     # 🔧 round < 40 儲存 safe copy
#     if current_round < 40:
#         if not hasattr(global_model, "safe_copy"):
#             global_model.safe_copy = {}
#         global_model.safe_copy[class_id] = avg.detach().clone()

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     dist_old_to_avg = (avg - old_wb).norm()
#     dist_old_to_updated = (updated - old_wb).norm()
#     ratio_thresh = 2.5 if current_round >= 100 else 2.0
#     if dist_old_to_updated > dist_old_to_avg * ratio_thresh:
#         print(f"[WARNING] round={current_round} class={class_id} | updated 偏離過大，使用 avg")
#         updated = avg.detach().clone()

#     # 🔧 updated 崩潰 fallback
#     if updated.norm().item() < 1e-5 or torch.isnan(updated).any() or torch.isinf(updated).any() or updated.norm().item() > 300.0:
#         print(f"[FALLBACK] round={current_round} class={class_id} | updated 無效，回退 safe copy")
#         updated = global_model.safe_copy.get(class_id, avg.detach().clone())

#     if not hasattr(global_model, "rolling_stats"):
#         global_model.rolling_stats = {
#             "acc_history": [],
#             "last_updated_class_params": {},
#             "last_valid_round": -1,
#             "panic_mode": False
#         }

#     global_model.rolling_stats["last_updated_class_params"][class_id] = old_wb.detach().clone()
#     global_model.rolling_stats["last_valid_round"] = current_round

#     if global_model.rolling_stats.get("panic_mode", False):
#         print(f"[PANIC] round={current_round} class={class_id} | 準確率崩潰，回復上次參數")
#         return global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())

#     print(f"[DEBUG] round={current_round:3d} | clients={values.size(0)} | ∆norm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | α={alpha:.2f} | avg.norm={avg_norm:.4f} | avg.mean={avg.mean().item():.4f}")
    
#     # 🔍 Acc-based panic detection
#     if hasattr(global_model, "rolling_stats"):
#         acc_hist = global_model.rolling_stats.get("acc_history", [])
#         if len(acc_hist) >= 6:
#             recent_avg = np.mean(acc_hist[-3:])
#             prev_avg = np.mean(acc_hist[-6:-3])
#             if prev_avg > 0.3 and (recent_avg < 0.5 * prev_avg):
#                 print(f"[PANIC MODE] round={current_round} class={class_id} | Accuracy drop detected: {prev_avg:.4f} -> {recent_avg:.4f}")
#                 global_model.rolling_stats["panic_mode"] = True
#                 global_model.rolling_stats["panic_round"] = current_round
#                 global_model.rolling_stats["panic_class_id"] = class_id
#                 updated = global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())
#                 return updated

#     # ======= 模型穩定性偵測 & Early Stopping Trigger =======
#     if not hasattr(global_model, "no_learning_counter"):
#         global_model.no_learning_counter = 0
#     if not hasattr(global_model, "training_should_stop"):
#         global_model.training_should_stop = False

#     delta_change = delta.norm().item()
#     avg_change = avg.norm().item() if class_id in global_model.history_avg_norms else 0.0

#     # 判斷 norm 是否穩定
#     def is_norm_stable(norm_seq: list, eps: float = 1e-3):
#         return len(norm_seq) >= 5 and all(abs(norm_seq[i] - norm_seq[i - 1]) < eps for i in range(1, len(norm_seq)))

#     avg_stable = is_norm_stable(global_model.history_avg_norms[class_id])
#     delta_stable = is_norm_stable(global_model.history_delta_norms[class_id])

#     if avg_stable and delta_stable:
#         global_model.no_learning_counter += 1
#         print(f"[STABILITY] round={current_round} class={class_id} | no_learning_counter={global_model.no_learning_counter}")
#     else:
#         global_model.no_learning_counter = 0  # reset if unstable

#     # 若連續穩定超過 patience 輪，設定停止訓練
#     patience = 10
#     if global_model.no_learning_counter >= patience:
#         print(f"[EARLY STOPPING TRIGGERED] round={current_round} class={class_id} | 模型已穩定，建議中止訓練")
#         global_model.training_should_stop = True
        
#     return updated


#---------------------------------
#early stop版本v2--->目前可以C8E4跑完500Round(運氣好)
# def weighted_avg_with_momentum_ACG(
#     values: Union[list[torch.Tensor], torch.Tensor],
#     weights: list[float],
#     class_id: int,
#     global_model: torch.nn.Module,
#     momentum_beta: float = 0.9,
#     current_round: int = 0,
#     boost_interval: int = 70,
#     min_clients_threshold: int = 7,
#     momentum_exit_round: int = 125,
#     local_epoch: int = 4
# ) -> torch.Tensor:
#     device = global_model.logits.weight.device
#     num_clients = len(weights)

#     if isinstance(values, list):
#         values = torch.stack(values).to(device)
#     else:
#         values = values.to(device)
#         if values.dim() == 1:
#             values = values.unsqueeze(0)

#     weights = torch.tensor(weights, dtype=torch.float32, device=device)
#     weights = weights / weights.sum()
#     weights = weights.view(-1, 1)
#     avg = torch.sum(values * weights, dim=0)

#     weight_key = f"logits.weight.{class_id}"
#     bias_key = f"logits.bias.{class_id}"

#     if hasattr(global_model, "global_momentum"):
#         lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
#         lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
#         old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
#     else:
#         old_wb = torch.cat([global_model.logits.weight[class_id], global_model.logits.bias[class_id].unsqueeze(0)])

#     avg_norm_val = avg.norm()
#     norm_limit = 100.0
#     if avg_norm_val > norm_limit:
#         shrink_factor = norm_limit / (avg_norm_val + 1e-6)
#         avg = avg * shrink_factor + old_wb * (1 - shrink_factor)

#     if values.size(0) < min_clients_threshold and local_epoch >= 4:
#         if not hasattr(global_model, "history_avg"):
#             global_model.history_avg = {}
#         if class_id not in global_model.history_avg:
#             global_model.history_avg[class_id] = avg.detach().clone()
#         else:
#             prev_avg = global_model.history_avg[class_id]
#             diff = avg - prev_avg
#             diff_norm = diff.norm().item()
#             prev_norm = prev_avg.norm().item()
#             if prev_norm > 0 and diff_norm / prev_norm > 0.2:
#                 smoothing_factor = 0.5
#                 avg = prev_avg + smoothing_factor * diff
#             global_model.history_avg[class_id] = avg.detach().clone()

#     avg_norm = avg.norm().item()
#     if not hasattr(global_model, "history_norms"):
#         global_model.history_norms = []
#     global_model.history_norms.append(avg_norm)
#     if len(global_model.history_norms) > 15:
#         global_model.history_norms.pop(0)

#     min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
#     max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0
#     dynamic_range = max_avg_norm - min_avg_norm
#     adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

#     # 🔍 提早偵測條件 1：delta 波動大 + client 太少 + local epoch 太多
#     if (
#         values.size(0) < min_clients_threshold and
#         local_epoch >= 4 and
#         delta.norm().item() > 3.0 * np.mean(global_model.history_delta_norms.get(class_id, [1.0]))
#     ):
#         print(f"[EARLY EXIT] round={current_round} class={class_id} | delta 太大且 client 太少，跳過聚合")
#         return old_wb.detach().clone()
        
#     early_stage_momentum = (
#         values.size(0) < min_clients_threshold and
#         current_round < momentum_exit_round and
#         local_epoch >= 4
#     )

#     if values.size(0) < min_clients_threshold and not early_stage_momentum:
#         print(f"[INFO] round={current_round} class={class_id} | client 太少（{values.size(0)}），跳過 momentum")
#         return avg.detach().clone()

#     if avg_norm < 1e-3 or math.isnan(avg_norm) or math.isinf(avg_norm):
#         print(f"[WARNING] round={current_round} class={class_id} | avg 無效，fallback to old_wb")
#         return old_wb.detach().clone()

#     delta = avg - old_wb
#     if values.size(0) < min_clients_threshold and (torch.any(torch.isnan(delta)) or torch.any(torch.isinf(delta)) or delta.norm() < 1e-6):
#         print(f"[WARNING] round={current_round} class={class_id} | delta 無效，使用 avg")
#         return avg.detach().clone()

#     if not hasattr(global_model, "history_avg_norms"):
#         global_model.history_avg_norms = {}
#     if not hasattr(global_model, "history_delta_norms"):
#         global_model.history_delta_norms = {}

#     global_model.history_avg_norms.setdefault(class_id, []).append(avg.norm().item())
#     global_model.history_delta_norms.setdefault(class_id, []).append(delta.norm().item())
#     if len(global_model.history_avg_norms[class_id]) > 10:
#         global_model.history_avg_norms[class_id].pop(0)
#     if len(global_model.history_delta_norms[class_id]) > 10:
#         global_model.history_delta_norms[class_id].pop(0)

#     def is_stable(seq: list, eps=1e-3):
#         return len(seq) >= 5 and all(abs(seq[i] - seq[i - 1]) < eps for i in range(1, len(seq)))
    
#     if is_stable(global_model.history_avg_norms[class_id]) and is_stable(global_model.history_delta_norms[class_id]):
#         noise = torch.randn_like(avg) * 0.01
#         avg = avg + noise
#         print(f"[NOISE] round={current_round} class={class_id} | learning stagnation detected, noise injected.")
#         delta = avg - old_wb

#     # ✅ Early Stop 條件：早期輪次，客戶端太少或 epoch 太淺
#     if values.size(0) <= min_clients_threshold and local_epoch <= 4:
#         print(f"[EARLY STOP] round={current_round} class={class_id} | client={values.size(0)}, epoch={local_epoch} 條件過早，跳過")
#         return old_wb.detach().clone()

#     #此部分，常數應寫為公式，公式的部分參數要去使用遞增或是遞減，使其可以去解釋
#     client_scale_ratio = min(1.0, max(0.0, (num_clients - 2) / 18))
#     scale_factor = 0.1 + 0.6 * client_scale_ratio
#     base_beta = 0.95 - 0.15 * client_scale_ratio
#     delta_clip = 2.0 + 6.0 * client_scale_ratio
#     max_mom_norm = 10.0 + 40.0 * client_scale_ratio

#     if early_stage_momentum:
#         momentum_beta = max(0.5, min(0.95, base_beta * (1.0 - current_round / (momentum_exit_round + 1e-6))))
#     else:
#         momentum_beta = min(0.95, base_beta)

#     local_epoch_scale = max(0.6, 1.0 / (1.0 + 0.2 * (local_epoch - 1)))
#     delta_clip *= local_epoch_scale
#     scale_factor *= local_epoch_scale

#     # 🔧 新增 delta 穩定性偵測與自動降火機制
#     if not hasattr(global_model, "delta_stability"):
#         global_model.delta_stability = {}
#     global_model.delta_stability.setdefault(class_id, []).append(delta.norm().item())
#     if len(global_model.delta_stability[class_id]) > 5:
#         global_model.delta_stability[class_id].pop(0)

#     def delta_is_unstable(seq, threshold=1.5):
#         if len(seq) < 3:
#             return False
#         diffs = [abs(seq[i] - seq[i - 1]) for i in range(1, len(seq))]
#         return any(d > threshold * np.mean(seq) for d in diffs)

#     if delta_is_unstable(global_model.delta_stability[class_id]):
#         print(f"[STABILIZER] round={current_round} class={class_id} | Delta 波動過大，進行降火")
#         scale_factor *= 0.5
#         alpha = 0.3  # 明確降火 alpha
#         momentum_beta *= 0.8

#     delta_norm_val = delta.norm()
#     if delta_norm_val > delta_clip:
#         delta = delta * (delta_clip / (delta_norm_val + 1e-6))
#     delta = delta * scale_factor

#     if not hasattr(global_model, "logits_momentum"):
#         global_model.logits_momentum = {}
#     if class_id not in global_model.logits_momentum:
#         global_model.logits_momentum[class_id] = torch.zeros_like(delta)

#     prev_momentum = global_model.logits_momentum[class_id]
#     new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta
#     mom_norm = new_momentum.norm()

#     if torch.any(torch.isnan(new_momentum)) or torch.any(torch.isinf(new_momentum)):
#         print(f"[WARNING] round={current_round} class={class_id} | momentum 出錯，使用 prev")
#         new_momentum = prev_momentum.detach().clone()
#     elif mom_norm > max_mom_norm:
#         print(f"[CLIP] round={current_round} class={class_id} | momentum norm ({mom_norm:.4f}) > {max_mom_norm}, 進行 clip")
#         new_momentum = new_momentum * (max_mom_norm / (mom_norm + 1e-6))

#     global_model.logits_momentum[class_id] = new_momentum

#     avg_norm_ratio = min(1.0, adjusted_max_avg_norm / (avg_norm + 1e-6))
#     alpha_base = 1 / (1 + math.exp(-0.03 * (current_round - 60))) * 0.8
#     alpha = alpha_base * avg_norm_ratio
#     #--------------avg_norm若是太大，可做early stop跳過聚合
#     updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

#     # 🔧 round < 40 儲存 safe copy
#     if current_round < 40:
#         if not hasattr(global_model, "safe_copy"):
#             global_model.safe_copy = {}
#         global_model.safe_copy[class_id] = avg.detach().clone()

#     if not hasattr(global_model, "global_momentum"):
#         global_model.global_momentum = {}
#     global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
#     global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

#     dist_old_to_avg = (avg - old_wb).norm()
#     dist_old_to_updated = (updated - old_wb).norm()

#     ratio_thresh = 2.5 if current_round >= 100 else 2.0
#     if dist_old_to_updated > dist_old_to_avg * ratio_thresh:
#         print(f"[WARNING] round={current_round} class={class_id} | updated 偏離過大，使用 avg")
#         updated = avg.detach().clone()

#     # 🔧 updated 崩潰 fallback
#     if updated.norm().item() < 1e-5 or torch.isnan(updated).any() or torch.isinf(updated).any() or updated.norm().item() > 300.0:
#         print(f"[FALLBACK] round={current_round} class={class_id} | updated 無效，回退 safe copy")
#         updated = global_model.safe_copy.get(class_id, avg.detach().clone())

#     if not hasattr(global_model, "rolling_stats"):
#         global_model.rolling_stats = {
#             "acc_history": [],
#             "last_updated_class_params": {},
#             "last_valid_round": -1,
#             "panic_mode": False
#         }

#     global_model.rolling_stats["last_updated_class_params"][class_id] = old_wb.detach().clone()
#     global_model.rolling_stats["last_valid_round"] = current_round

#     if global_model.rolling_stats.get("panic_mode", False):
#         print(f"[PANIC] round={current_round} class={class_id} | 準確率崩潰，回復上次參數")
#         return global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())

#     print(f"[DEBUG] round={current_round:3d} | clients={values.size(0)} | ∆norm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | α={alpha:.2f} | avg.norm={avg_norm:.4f} | avg.mean={avg.mean().item():.4f}")
    
#     # 🔍 Acc-based panic detection
#     if hasattr(global_model, "rolling_stats"):
#         acc_hist = global_model.rolling_stats.get("acc_history", [])
#         if len(acc_hist) >= 6:
#             recent_avg = np.mean(acc_hist[-3:])
#             prev_avg = np.mean(acc_hist[-6:-3])
#             if prev_avg > 0.3 and (recent_avg < 0.5 * prev_avg):
#                 print(f"[PANIC MODE] round={current_round} class={class_id} | Accuracy drop detected: {prev_avg:.4f} -> {recent_avg:.4f}")
#                 global_model.rolling_stats["panic_mode"] = True
#                 global_model.rolling_stats["panic_round"] = current_round
#                 global_model.rolling_stats["panic_class_id"] = class_id
#                 updated = global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())
#                 return updated

#     return updated


#----------------
# 目前論文實用版
# 結合動量機制與多種安全防護條件來穩定地執行多客戶端的參數加權聚合
def weighted_avg_with_momentum_ACG(
    values: Union[list[torch.Tensor], torch.Tensor],
    weights: list[float],
    class_id: int,
    global_model: torch.nn.Module,
    momentum_beta: float = 0.9,
    current_round: int = 0,
    boost_interval: int = 70,
    min_clients_threshold: int = 7,
    momentum_exit_round: int = 125,
    local_epoch: int = 4
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
    
    #step 2. Shrink 與 fallback 處理：控制過大範數與異常數值（如 NaN / Inf）導致的模型爆炸問題。
    # ===== [新增] avg 防爆處理 - value clipping =====
    max_avg_val = 20.0  # 可調參數，預防過大激活值
    avg = torch.clamp(avg, -max_avg_val, max_avg_val)

    # ===== [新增] avg 防爆處理 - avg norm 二次檢查 =====
    avg_norm_val = avg.norm()

    # ===== [強化版 avg shrink，控制 norm 至 ~30] =====
    # 這段程式碼的目的是確保在模型更新過程中，對於不穩定或過大的參數變動進行控制。具體來說，會根據範數（norm）的值來進行縮放、回退操作，避免過度或無效的更新。
    #  計算目標範數並準備權重和偏置
    # 定一個目標範數 target_norm，用於控制更新的大小，避免過大的更新。
    # 構建 weight_key 和 bias_key，用於提取指定 class_id 的權重和偏置。
    # 根據是否存在 global_momentum，選擇使用提前計算的動量（lookahead_weight 和 lookahead_bias）還是當前的權重和偏置。
    target_norm = 30.0
    weight_key = f"logits.weight.{class_id}"
    bias_key = f"logits.bias.{class_id}"

    if hasattr(global_model, "global_momentum"):
        lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
        lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
        old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
    else:
        old_wb = torch.cat([global_model.logits.weight[class_id], global_model.logits.bias[class_id].unsqueeze(0)])

    # 範數檢查與縮放操作：
    # 這部分對 avg 的範數進行檢查。如果 avg.norm() 超過了目標範數 target_norm 且客戶端數量少於 min_clients_threshold，則會進行縮放操作。
    # 透過縮放因子 shrink_factor 來減小 avg 的範數，使其不超過目標範數，並將縮放後的 avg 與舊的 old_wb 進行混合。
    original_norm = avg.norm().item()
    if original_norm > target_norm and values.size(0) < min_clients_threshold:
        shrink_factor = target_norm / (original_norm + 1e-6)
        avg = avg * shrink_factor + old_wb * (1 - shrink_factor)
        updated_norm = avg.norm().item()
        # 最後，程式會打印出縮放前後的範數變化。
        print(f"[SHRINK] round={current_round} class={class_id} | avg.norm 過高: {original_norm:.3f} > {target_norm}，進行 shrink")
        print(f"         shrink factor={shrink_factor:.4f} | 更新後 avg.norm={updated_norm:.3f}")
    
    # 檢查無效或爆炸的範數：
    # 如果 avg 的範數是 NaN 或 Inf，或範數過大（超過 200.0），則視為更新無效或爆炸，程式會回退至舊的 old_wb，以保證訓練過程不會受到無效更新的影響。
    if math.isnan(avg_norm_val) or math.isinf(avg_norm_val) or avg_norm_val > 200.0:
        print(f"[FALLBACK] round={current_round} class={class_id} | avg.norm={avg_norm_val:.4f} 無效或爆炸，回退至 old_wb")
        return old_wb.detach().clone()
    
    # 如果 old_wb 本身就包含了 NaN 或 Inf，則會打印錯誤信息，表示該權重包含無效數據，這樣的情況會導致訓練無法正常進行。
    if torch.isnan(old_wb).any() or torch.isinf(old_wb).any():
        print(f"[ERROR] old_wb 本身已經含有 NaN 或 Inf！")

    # step 3. 歷史平均平滑處理（smoothing）：當 client 太少或更新不穩定時，透過歷史平均平滑 avg。
    # 這段程式碼的目的是對 avg（更新後的權重平均）進行正規化處理，並在特定條件下使用歷史平均來平滑更新，避免過度的變動或誤差。
    # 正規化 avg
    # avg.norm() 計算了 avg 的範數（即它的模長）。
    # 如果 avg 的範數超過了設定的 norm_limit（此處為 100.0），則對 avg 進行縮放處理，確保它不會超過範圍。縮放比例 shrink_factor 會根據當前範數與限制範數的比值計算。
    # 這樣做的目的是防止 avg 的更新變得過大，從而導致模型不穩定。
    avg_norm_val = avg.norm()
    norm_limit = 100.0
    if avg_norm_val > norm_limit:
        shrink_factor = norm_limit / (avg_norm_val + 1e-6)
        avg = avg * shrink_factor + old_wb * (1 - shrink_factor)

     # step 4. 動態 norm 控制：根據輪次歷史範數，自適應地設定 norm 限制。
    # 歷史平均更新
    # 當 values.size(0) 小於 min_clients_threshold(最小客戶端容忍數) 且 local_epoch >= 4 時，程式開始使用歷史的 avg 進行平滑處理。
    # global_model.history_avg 用來儲存每個 class_id 的歷史平均。
    # 如果該 class_id 還沒有對應的歷史平均，則將當前的 avg 存入並設為 prev_avg
    if values.size(0) < min_clients_threshold and local_epoch >= 4:
        if not hasattr(global_model, "history_avg"):
            global_model.history_avg = {}
        if class_id not in global_model.history_avg:
            global_model.history_avg[class_id] = avg.detach().clone()
            prev_avg = avg.detach().clone()  # 🔧 fallback 定義
        else:
            prev_avg = global_model.history_avg[class_id]
            diff = avg - prev_avg
            diff_norm = diff.norm().item()
            prev_norm = prev_avg.norm().item()
            if prev_norm > 0 and diff_norm / prev_norm > 0.2:
                smoothing_factor = 0.5
                # 否則，程式計算 avg 和 prev_avg 之間的差異（diff），並且根據差異的範數（diff_norm）與歷史平均範數（prev_norm）來決定是否進行平滑處理。
                # 若差異過大，則進行加權平滑，將 avg 更新為 prev_avg 與差異的加權和。
                avg = prev_avg + smoothing_factor * diff
            # 更新後的 avg 存回 history_avg 中，保證下一次能夠使用最新的歷史平均。
            global_model.history_avg[class_id] = avg.detach().clone()
    else:
        # 如果 values.size(0) 不小於 min_clients_threshold 或 local_epoch 小於 4，則簡單地將當前的 avg 儲存為 prev_avg，用於後續的計算或回退。
        prev_avg = avg.detach().clone()  # 🔧 fallback for downstream use

    # avg_norm 計算的是 avg 向量的範數（即它的模長），通常用來衡量模型權重的大小。這是後續調整更新的關鍵數據。
    avg_norm = avg.norm().item()
    # 存儲歷史的 avg_norm
    # 這段程式碼會將當前的 avg_norm 存儲到 global_model.history_norms 列表中，以便跟踪過去 15 次更新中的範數變化。
    # 如果歷史範數列表的長度超過 15，會刪除最舊的範數值，保持列表長度為 15。
    if not hasattr(global_model, "history_norms"):
        global_model.history_norms = []
    global_model.history_norms.append(avg_norm)
    if len(global_model.history_norms) > 15:
        global_model.history_norms.pop(0)

    # 計算歷史範數的最小值和最大值
    # min_avg_norm 和 max_avg_norm 分別表示歷史範數中的最小值和最大值。如果歷史範數列表的長度小於等於 1，則預設最小值為 10.0，最大值為 50.0。
    # 這是為了防止在初期階段，範數列表過短導致的極端值影響。
    min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
    max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0

    # 計算動態範圍（dynamic_range）並調整最大平均範數
    # dynamic_range 計算了歷史範數的範圍，即最大值與最小值之間的差距。
    # adjusted_max_avg_norm 是根據歷史範數範圍動態調整的最大範數。隨著訓練輪次的增長（current_round），adjusted_max_avg_norm 會逐漸增大，但不會超過原來的範圍。
    dynamic_range = max_avg_norm - min_avg_norm
    adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

    # step 5. 動量條件與 early stage momentum 判斷：在特定條件下啟用 momentum（動量）更新。
    # 判斷是否進入 early_stage_momentum（早期階段的動量更新）
    # early_stage_momentum 的邏輯： early_stage_momentum 用來控制在客戶端數量較少、訓練輪次較少的情況下，是否啟用動量更新。這樣的設計有助於在早期階段穩定訓練過程，避免過早引入過多的動量，從而影響模型的更新。
    early_stage_momentum = (
        values.size(0) < min_clients_threshold and
        current_round < momentum_exit_round and
        local_epoch >= 4
    )
    # 如果客戶端數量少於 min_clients_threshold，並且不處於早期階段的動量更新（early_stage_momentum 為 False），則會跳過動量更新。
    if values.size(0) < min_clients_threshold and not early_stage_momentum:
        print(f"[INFO] round={current_round} class={class_id} | client 太少（{values.size(0)}），跳過 momentum")
        return avg.detach().clone()
    
    # 如果 avg_norm（即平均範數）小於 1e-3，或者是 NaN 或 inf（即無效數值），則會打印警告並回退到先前的權重 old_wb。這樣做是為了防止使用無效的更新來更新模型。
    if avg_norm < 1e-3 or math.isnan(avg_norm) or math.isinf(avg_norm):
        print(f"[WARNING] round={current_round} class={class_id} | avg 無效，fallback to old_wb")
        return old_wb.detach().clone()

    # 如果 avg_norm 是有效的，並且進入了動量更新階段，則計算 delta，即當前平均權重 avg 和先前權重 old_wb 之間的差異。這個 delta 代表了權重的變化量，通常在動量更新中會被用來調整模型參數。
    delta = avg - old_wb

    # if not hasattr(global_model, "history_avg_norms"):
    #     global_model.history_avg_norms = {}
    # if not hasattr(global_model, "history_delta_norms"):
    #     global_model.history_delta_norms = {}

    # global_model.history_avg_norms.setdefault(class_id, []).append(avg.norm().item())
    # global_model.history_delta_norms.setdefault(class_id, []).append(delta.norm().item())
    # if len(global_model.history_avg_norms[class_id]) > 10:
    #     global_model.history_avg_norms[class_id].pop(0)
    # if len(global_model.history_delta_norms[class_id]) > 10:
    #     global_model.history_delta_norms[class_id].pop(0)

    # #此方法avg.norm() 的變化趨於收斂時，自動跳過這次的聚合
    # def is_converged(seq: list, epsilon=0.01, window=5):
    #     return len(seq) >= window and all(abs(seq[i] - seq[i - 1]) < epsilon for i in range(-window + 1, 0))

    # if (avg.norm().item() - prev_avg.norm().item()) > 10.0 and values.size(0) < min_clients_threshold:
    #     print(f"[FALLBACK] round={current_round} class={class_id} | avg.norm 相對上一輪超出10，使用上一輪參數")
    #     return old_wb.detach().clone()

    # if is_converged(global_model.history_avg_norms[class_id]) and values.size(0) < min_clients_threshold:
    #     print(f"[SKIP] round={current_round} class={class_id} | avg.norm 收斂，跳過聚合")
    #     return old_wb.detach().clone()

    # step 6. 動態調整 scale_factor、momentum beta、delta clip 限制：隨 client 數與輪次自動調整參數強度與更新範圍。
    # 此段的目的是根據客戶端數量來動態調整不同的訓練參數
    raw_ratio = (num_clients - 2) / 18
    # client_scale_ratio - 描述目前客戶端數在合理區間內的比例（最多 0.778），作為調整參數比例的主控變數。(當 client 多時，允許更 激進 的參數（如放大更新、降低動量慣性、擴大 clip 上限等）)
    client_scale_ratio = min(0.778, max(0.0, raw_ratio))
    # scale_factor - 對 delta 的縮放倍率，根據 client_scale_ratio 與 local_epoch_scale 調整，用來控制更新強度。(起始值 0.1：是保守學習的底線，代表只有 0.1 的學習強度。最大值約為 0.1 + 0.6 * 0.778 ≈ 0.567，不到 1 是因為聯邦學習偏保守（非中央化訓練）)
    scale_factor = 0.1 + 0.6 * client_scale_ratio
    # base_beta - Momentum 基礎 beta 值（衰減比率），根據 client_scale_ratio 調整，決定歷史與當前權重的比。(base_beta 控制 momentum 的衰減速率。在 client 多的情況下，減少 beta（= 0.95 - ...），使動量更加「即時」反應新梯度。0.95 為 momentum 的經典預設值（如 Adam）)，減少最多 0.15，是我設定為「最多降到 0.8」的安全範圍（超過這值動量會太跳）
    base_beta = 0.95 - 0.15 * client_scale_ratio
    # delta_clip - 對 delta 的 norm 限制上限，避免一次變化過大，依據 client 數與 local_epoch 調整。(低 client 時使用較低 clip（僅允許小幅變動），高 client 時放寬)
    delta_clip = 2.0 + 6.0 * client_scale_ratio
    # max_mom_norm - 限制 momentum 的最大 norm，若超出則進行 clip，確保動量不會爆炸。(大部分 norm clip 的實踐中（如 gradient norm clipping），10 是常見上限的起始設計。為何乘 40？這個斜率決定調整幅度，40 是設計者依據實驗經驗，找到的「足夠大但穩定」的增幅)
    max_mom_norm = 10.0 + 40.0 * client_scale_ratio
    # log 參數至 wandb（用 round 作為 step）
    wandb.log({
        "client_scale_ratio": client_scale_ratio,
        "scale_factor": scale_factor,
        "base_beta": base_beta,
        "delta_clip": delta_clip,
        "max_mom_norm": max_mom_norm,
    }, step=current_round)

    # 動態調整 momentum 參數與「對 delta 的修正與縮放」，以提升訓練穩定性並適應當前的訓練狀況。
    # 調整 momentum 的 momentum_beta

    # early_stage_momentum 為 True 時， 根據當前訓練輪數（current_round）來動態調整 momentum_beta
    # 隨著訓練進行，momentum_beta 逐漸減小（從較大的 base_beta 開始），這有助於在訓練初期強化 momentum，並隨著輪數增加逐步減少 momentum 的影響，使得後期更新更加平穩。
    # 計算公式將 momentum_exit_round（大概是設定的一個結束輪數）用來控制 momentum 的減少速度，避免 momentum 在早期過於強大。
    if early_stage_momentum:
        momentum_beta = max(0.5, min(0.95, base_beta * (1.0 - current_round / (momentum_exit_round + 1e-6))))
    #若訓練進入後期（early_stage_momentum = False），則保持 momentum 在 0.95 以下，確保不過度干擾。
    else:
        momentum_beta = min(0.95, base_beta)

    # 動態縮放 local_epoch_scale，調整 delta 和 scale_factor：
    # 隨著訓練輪數的增加，local_epoch_scale 用來動態調整訓練的「學習率」或「更新幅度」，它是根據當前的本地 epoch 數（local_epoch）來縮放的：
    # local_epoch_scale 的計算公式是使得隨著 local_epoch 的增大，縮放因子逐步減少。避免過早的過大更新，並且能在後期進行更精細的調整。
    # delta_clip 和 scale_factor 會乘以這個縮放因子，使得更新更加穩定並根據訓練進程逐漸減少步伐。
    local_epoch_scale = max(0.6, 1.0 / (1.0 + 0.2 * (local_epoch - 1)))
    delta_clip *= local_epoch_scale
    scale_factor *= local_epoch_scale

    #  對 delta 進行裁剪：
    # 這裡計算了 delta（即梯度或更新量）的 L2 norm，並對 delta 進行裁剪，防止過大的更新。
    # 若 delta 的 norm 超過設定的 delta_clip，則將其縮放，使其 norm 恢復到 delta_clip 的範圍內，從而防止梯度爆炸。
    delta_norm_val = delta.norm()
    if delta_norm_val > delta_clip:
        delta = delta * (delta_clip / (delta_norm_val + 1e-6))
    #最後對 delta 進行縮放，最後根據之前的縮放因子 scale_factor 對 delta 進行縮放，使得更新的幅度適應訓練的狀況，避免過早的過大更新影響到模型的訓練。    
    delta = delta * scale_factor
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
    prev_momentum = global_model.logits_momentum[class_id]
    new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta

    # 防呆檢查與 clipping
    mom_norm = new_momentum.norm()

    # 若 new_momentum 出現 NaN 或 inf，就使用 prev_momentum 取代，避免崩潰。
    if torch.any(torch.isnan(new_momentum)) or torch.any(torch.isinf(new_momentum)):
        print(f"[WARNING] round={current_round} class={class_id} | momentum 出錯，使用 prev")
        new_momentum = prev_momentum.detach().clone()
    # 若其 L2 norm 太大，則進行 clip：防止更新過猛，避免模型爆炸。
    elif mom_norm > max_mom_norm:
        print(f"[CLIP] round={current_round} class={class_id} | momentum norm ({mom_norm:.4f}) > {max_mom_norm}, 進行 clip")
        new_momentum = new_momentum * (max_mom_norm / (mom_norm + 1e-6))
    # 儲存更新後的 momentum
    global_model.logits_momentum[class_id] = new_momentum

    # 自適應調整 blending factor alpha
    # alpha 是 logits 更新中，momentum 權重的參數：
    # avg_norm_ratio 控制更新幅度（norm 太大就降低比例）。
    # alpha_base 使用 sigmoid 控制，60 輪前漸進增加。
    # alpha 越大代表「越相信 momentum」。
    avg_norm_ratio = min(1.0, adjusted_max_avg_norm / (avg_norm + 1e-6))
    alpha_base = 1 / (1 + math.exp(-0.03 * (current_round - 60))) * 0.8
    alpha = alpha_base * avg_norm_ratio
    # updated 是最終版本，送回去作為新的 logits 層參數。
    # 最終 logits 層參數更新：混合使用兩個資訊：1.avg：整體平均參數（較穩定）。 2.old_wb + new_momentum：加權的慣性更新方向（包含當前變化趨勢）。
    updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

    # 🔧 round < 40 儲存 safe copy
    # 在前 40 輪內儲存 avg 當作每個 class 的 safe_copy
    if current_round < 40:
        if not hasattr(global_model, "safe_copy"):
            global_model.safe_copy = {}
        global_model.safe_copy[class_id] = avg.detach().clone()

    # 這段將 momentum 的權重部分和 bias 部分分別儲存起來，方便後續模型更新使用。
    if not hasattr(global_model, "global_momentum"):
        global_model.global_momentum = {}
    global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
    global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()
    # 這段是「防止更新偏離過大」的防呆機制。
    # 比較：
    # avg：正常平均值。
    # updated：目前本輪的實際更新結果。
    # old_wb：上一輪的模型參數。

    # step 7. 比較 updated 與過去平均參數 avg 的距離是否偏離太大
    # 比較距離：
    # 若 updated 與過去的距離遠超過 avg 的距離上限倍數（預設 2.0 到 2.5 倍），則表示偏離過大。
    # 將 updated 直接改為 avg.detach().clone()，強制退回較穩定的平均版本。
    dist_old_to_avg = (avg - old_wb).norm()
    dist_old_to_updated = (updated - old_wb).norm()
    ratio_thresh = 2.5 if current_round >= 100 else 2.0
    if dist_old_to_updated > dist_old_to_avg * ratio_thresh:
        print(f"[WARNING] round={current_round} class={class_id} | updated 偏離過大，使用 avg")
        updated = avg.detach().clone()

    # step 8. 檢查 updated 是否出現數值異常，啟動 fallback
    # # 🔧 updated 崩潰 fallback
    # 判斷該更新結果是否「數值異常」或「崩潰」。
    # updated.norm().item() < 1e-5Norm 非常接近 0，表示幾乎沒有學習到東西（可能是某些 gradient vanishing 的情況）。
    # torch.isnan(updated).any()：檢查是否有 NaN（非數值）存在，這是數值爆炸或訓練不穩的警訊。
    # torch.isinf(updated).any()：檢查是否有無限值（Inf、-Inf），這會導致後續模型崩潰。
    # updated.norm().item() > 300.0：Norm 過大，可能代表梯度爆炸、模型不穩定。
    # 任何「數值異常」的更新都會被視為不安全，系統會採取 回退機制（fallback）。
    if updated.norm().item() < 1e-5 or torch.isnan(updated).any() or torch.isinf(updated).any() or updated.norm().item() > 300.0:
        print(f"[FALLBACK] round={current_round} class={class_id} | updated 無效，回退 safe copy")
        updated = global_model.safe_copy.get(class_id, avg.detach().clone())

    # step 9. 初始化 rolling_stats 結構（若尚未存在）
    # rolling_stats 是用來評估每次模型聚合之後的品質表現
    # "acc_history": [] 用來記錄每一輪訓練/聚合的準確率或相關表現指標，通常會 append 各輪結果。
    # "last_updated_class_params": {} 是一個 dict，key 是 class_id，value 是對應聚合成功後的模型參數 snapshot。
    # "last_valid_round": -1 用來表示目前尚未有「有效聚合」的紀錄（因為還沒開始訓練或還沒成功聚合過）。
    # "panic_mode": False 表示目前模型系統處於正常狀態，尚未偵測到異常（如：模型退化、準確率暴跌等）。
    if not hasattr(global_model, "rolling_stats"):
        global_model.rolling_stats = {
            "acc_history": [],
            "last_updated_class_params": {},
            "last_valid_round": -1,
            "panic_mode": False
        }
    
    # step 10. 更新每個 class 的穩定參數快照與最後有效聚合輪數
    # "last_updated_class_params"：記錄每個 class_id 上一次被「成功聚合」時的模型參數。
    # "last_valid_round"：紀錄最近一次「有效聚合」的訓練輪數。
    # 在模型訓練過程中，每一輪訓練（或聚合）結束後，只有在沒有觸發 panic 模式時，也就是聚合結果被判定為「可接受」的情況下才會被執行。
    global_model.rolling_stats["last_updated_class_params"][class_id] = old_wb.detach().clone()
    global_model.rolling_stats["last_valid_round"] = current_round

    #step 11. 若處於 panic 模式，強制使用上一版參數
    # 當模型進入 panic mode 時，執行參數回退，避免惡化的模型繼續學習與更新。
    # 從 rolling_stats 裡查詢 panic 模式是否被啟動（True），預設為 False。
    if global_model.rolling_stats.get("panic_mode", False):
        print(f"[PANIC] round={current_round} class={class_id} | 準確率崩潰，回復上次參數")
        return global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())

    #step 12. 列印當前回合所有關鍵變數的數值（debug 訊息）
    #以下是顯示各變數，每一回合聚合各client之梯度與動量等變數數值
    print(
        f"[DEBUG] round={current_round:3d} | clients={values.size(0)} | "
        f"∆norm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | "
        f"new_mom.norm={new_momentum.norm():.4f} | "
        f"updated.norm={updated.norm():.4f} | updated.mean={updated.mean().item():.4f} | "
        f"α={alpha:.2f} | base_β={base_beta:.2f} | "
        f"scale={scale_factor:.3f} | epoch_scale={local_epoch_scale:.3f} | "
        f"∆clip={delta_clip:.2f} | avg.norm={avg_norm:.4f} | avg.mean={avg.mean().item():.4f}"
    )

    # step 12. 偵測準確率是否異常下滑（Acc-based Panic Detection）
    # 🔍 Acc-based panic detection 偵測某類別模型準確率是否出現「異常下滑」，並在此情況下啟動恐慌模式（panic mode）
    if hasattr(global_model, "rolling_stats"):
        # 從 rolling_stats 中取得準確率歷史紀錄列表 acc_history，代表某類別歷輪的準確率表現。
        acc_hist = global_model.rolling_stats.get("acc_history", [])
        # 只有當準確率歷史紀錄至少有 6 筆資料時，才進行分析，確保資料量足夠判斷趨勢變化。
        if len(acc_hist) >= 6:
            # 將最近 3 輪的準確率 (recent_avg) 與再之前 3 輪的準確率 (prev_avg) 做平均，用來比對變化趨勢。
            recent_avg = np.mean(acc_hist[-3:])
            prev_avg = np.mean(acc_hist[-6:-3])
            # 若歷史表現還不錯（prev_avg 大於 0.3），但最近表現大幅下滑（recent_avg 少於之前的一半），即視為異常下降 → 觸發恐慌模式。
            if prev_avg > 0.3 and (recent_avg < 0.5 * prev_avg):
                # 輸出提示訊息，指出哪一輪與哪個類別出現了準確率崩盤。
                print(f"[PANIC MODE] round={current_round} class={class_id} | Accuracy drop detected: {prev_avg:.4f} -> {recent_avg:.4f}")
                global_model.rolling_stats["panic_mode"] = True
                global_model.rolling_stats["panic_round"] = current_round
                global_model.rolling_stats["panic_class_id"] = class_id
                # 在 rolling_stats 中標記目前進入了恐慌模式，並記錄觸發此狀況的回合與類別。
                updated = global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())
                # 返回「上一次更新成功的 class 參數」，也就是在恐慌模式下，不採用目前 round 的結果，而是回退使用前一版本參數，避免錯誤擴散。
                return updated
    
    # step 13. 回傳最終 updated 參數
    return updated