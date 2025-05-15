import torch
import copy
from data_preprocessing import Dataset
from models import model_train, model_train_FedProx, model_train_MOON, model_eval

# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Client(object):
    """
    Self-defined client class.
    """

    def __init__(self, args: object, client_name: str, client_data_dict: dict[str, torch.Tensor]) -> None:
        """
        Arguments:
            args (argparse.Namespace): parsed argument object.
            client_name (str): client name / id.
            client_data_dict (dict[str, torch.Tensor]): a dictionary holding all data of this client, with 'x' and 'y' as keys. 
        """

        super(Client, self).__init__()
        self.client_name   = client_name
        self.num_sample    = len(client_data_dict['y'])
        self.client_epoch  = args.client_epoch
        self.client_bs     = args.client_bs
        
        # for FedProx
        self.FedProx = args.FedProx

        # for MOON
        self.MOON = args.MOON
        
        # datasets and data loaders
        self.dataset = Dataset(client_data_dict['x'], client_data_dict['y'])
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size = self.client_bs, shuffle = not self.MOON, pin_memory = True)
            
    def local_train(self, client_model: torch.nn.Module, global_model: torch.nn.Module, previous_feature: torch.Tensor) -> list | torch.Tensor:
        """
        Client local training with lookahead initialization for faster learning.

        Arguments:
            client_model (torch.nn.Module): pytorch model (client local model).
            global_model (torch.nn.Module): pytorch model (global model).
            previous_feature (torch.Tensor): features extracted by client model in last global epoch, useful for MOON.

        Returns:
            last_client_features (list | torch.Tensor): empty list, or features extracted by client model in current global epoch.
        """
        print(f"[{self.client_name}] local_train() started.")
        client_model.to(device)

        # Step 1: Apply lookahead initialization to client model before training
        self.apply_lookahead_initialization(client_model, global_model)

        client_features = []
        if self.MOON:
            for current_client_epoch in range(self.client_epoch):
                # client model train
                if (previous_feature != None) and (client_features == []):
                    client_features_tensor = previous_feature
                elif (previous_feature == None) and (client_features == []):
                    client_features_tensor = None
                elif client_features != []:
                    client_features_tensor = torch.zeros((len(client_features), client_features[0].shape[0], client_features[0].shape[1]))
                    for idx, prev in enumerate(client_features):
                        client_features_tensor[idx] = copy.deepcopy(prev.detach())
                    client_features_tensor = client_features_tensor.cuda()

                client_feat = model_train_MOON(client_model, global_model, self.data_loader, client_features_tensor)
                client_features.append(client_feat)
        elif self.FedProx:
            model_train_FedProx(client_model, global_model, self.data_loader, self.client_epoch)
        else:
            model_train(client_model, self.data_loader, self.client_epoch)

        client_model.to('cpu')
        last_client_features = []
        if self.MOON:
            last_client_features = client_features[-1]

        return client_model, last_client_features  # ✅ return 修改後的 model
    # 這段程式碼的目的是將來自全局模型（global_model）的權重與動量（momentum）更新應用到本地模型（client_model）中，並且進行 Lookahead 初始化。
    # Lookahead 是一種策略，旨在通過引入全局的動量來加速訓練。

    def apply_lookahead_initialization(self, client_model: torch.nn.Module, global_model: torch.nn.Module) -> None:
        print(f"[{self.client_name}] apply_lookahead_initialization() start.")
        #  設置設備
        device = next(global_model.parameters()).device
        # 這段程式碼遍歷全局模型（global_model）中每個類別的權重（logits.weight）及偏置（logits.bias），對每個類別進行初始化。
        for class_id in range(len(global_model.logits.weight)):
            # 複製權重與偏置
            weight = global_model.logits.weight[class_id].data.clone().to(device)
            bias = global_model.logits.bias[class_id].data.clone().to(device)

            #  檢查並應用來自 global_momentum 的動量（Lookahead）
            # 如果全局模型（global_model）擁有 global_momentum 屬性，則檢查是否存在對應於當前類別的動量。動量會被加到原始的權重和偏置上，進行 "lookahead" 操作，從而加速模型更新。
            # lookahead from global_momentum if exists
            if hasattr(global_model, "global_momentum"):
                weight_key = f"logits.weight.{class_id}"
                bias_key = f"logits.bias.{class_id}"

                if weight_key in global_model.global_momentum:
                    weight += global_model.global_momentum[weight_key].to(device)
                if bias_key in global_model.global_momentum:
                    bias += global_model.global_momentum[bias_key].to(device)
            # 檢查並應用來自 logits_momentum 的動量
            # 如果 global_model 擁有 logits_momentum 且當前類別的動量存在，則將動量加到權重和偏置上。這樣不僅使用了全局的動量，還會加入當前本地的動量。
            # plus current logits_momentum if exists
            if hasattr(global_model, "logits_momentum") and class_id in global_model.logits_momentum:
                momentum = global_model.logits_momentum[class_id].to(device)
                weight += momentum[:-1]
                bias += momentum[-1]
                # print(f"[{self.client_name}] class {class_id}: logits_momentum norm = {momentum.norm().item():.4f}")
            # else:
                # print(f"[{self.client_name}] class {class_id}: no logits_momentum.")

            # 用更新過的 weight 和 bias 來替換客戶端模型（client_model）中對應類別的權重和偏置。
            # update client model
            client_model.logits.weight[class_id].data.copy_(weight)
            client_model.logits.bias[class_id].data.copy_(bias)

        # 在整個初始化過程完成後，輸出一條消息，表明 Lookahead 初始化已經結束。
        print(f"[{self.client_name}] apply_lookahead_initialization() done.")



    def local_eval(self, client_model: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        """
        (Obsolete.) Conduct inference locally.

        Arguments:
            client_model (torch.nn.Module): pytorch model (client local model).

        Returns:
            labels (torch.Tensor): ground truth labels.
            preds (torch.Tensor): logits (not softmaxed yet).
        """

        client_model.to(device)
        labels, preds = model_eval(client_model, self.data_loader, {}, '', True)
        client_model.to('cpu')
        return labels, preds


def get_clients(args: object, data_dict: dict[str, dict[str, torch.Tensor]]) -> list[Client]:
    """
    Intialize client objects using data dictionary.

    Arguments:
        args (argparse.Namespace): parsed argument object.
        data_dict (dict[str, dict[str, torch.Tensor]]): a dictionary that contains all data with user id as keys. Each value entry is also a dictionary with 'x', 'y' as keys and data tensor as values.

    Returns:
        clients (list[Client]): list of clients.
    """

    clients = []
    for client_name, client_data_dict in data_dict.items():
        client = Client(args, client_name, client_data_dict)
        clients.append(client)
    return clients
