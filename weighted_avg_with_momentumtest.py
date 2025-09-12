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
    device = global_model.logits.weight.device
    num_clients = len(weights)

    if isinstance(values, list):
        values = torch.stack(values).to(device)
    else:
        values = values.to(device)
        if values.dim() == 1:
            values = values.unsqueeze(0)

   
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    weights = weights / weights.sum()
    weights = weights.view(-1, 1)
    avg = torch.sum(values * weights, dim=0)

    max_avg_val = 20.0  # 可調參數，預防過大激活值
    avg = torch.clamp(avg, -max_avg_val, max_avg_val)

    avg_norm_val = avg.norm()

    target_norm = 30.0
    weight_key = f"logits.weight.{class_id}"
    bias_key = f"logits.bias.{class_id}"

    if hasattr(global_model, "global_momentum"):
        lookahead_weight = global_model.global_momentum.get(weight_key, global_model.logits.weight[class_id])
        lookahead_bias = global_model.global_momentum.get(bias_key, global_model.logits.bias[class_id])
        old_wb = torch.cat([lookahead_weight.to(device), lookahead_bias.unsqueeze(0).to(device)])
    else:
        old_wb = torch.cat([global_model.logits.weight[class_id], global_model.logits.bias[class_id].unsqueeze(0)])

    original_norm = avg.norm().item()
    if original_norm > target_norm and values.size(0) < min_clients_threshold:
        shrink_factor = target_norm / (original_norm + 1e-6)
        avg = avg * shrink_factor + old_wb * (1 - shrink_factor)
        updated_norm = avg.norm().item()
        # 最後，程式會打印出縮放前後的範數變化。
        print(f"[SHRINK] round={current_round} class={class_id} | avg.norm 過高: {original_norm:.3f} > {target_norm}，進行 shrink")
        print(f"         shrink factor={shrink_factor:.4f} | 更新後 avg.norm={updated_norm:.3f}")
    
    
    if math.isnan(avg_norm_val) or math.isinf(avg_norm_val) or avg_norm_val > 200.0:
        print(f"[FALLBACK] round={current_round} class={class_id} | avg.norm={avg_norm_val:.4f} 無效或爆炸，回退至 old_wb")
        return old_wb.detach().clone()
    
    # 如果 old_wb 本身就包含了 NaN 或 Inf，則會打印錯誤信息，表示該權重包含無效數據，這樣的情況會導致訓練無法正常進行。
    if torch.isnan(old_wb).any() or torch.isinf(old_wb).any():
        print(f"[ERROR] old_wb 本身已經含有 NaN 或 Inf！")

    avg_norm_val = avg.norm()
    norm_limit = 100.0
    if avg_norm_val > norm_limit:
        shrink_factor = norm_limit / (avg_norm_val + 1e-6)
        avg = avg * shrink_factor + old_wb * (1 - shrink_factor)

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
                avg = prev_avg + smoothing_factor * diff
            global_model.history_avg[class_id] = avg.detach().clone()
    else:
        # 如果 values.size(0) 不小於 min_clients_threshold 或 local_epoch 小於 4，則簡單地將當前的 avg 儲存為 prev_avg，用於後續的計算或回退。
        prev_avg = avg.detach().clone()  # 🔧 fallback for downstream use

    # avg_norm 計算的是 avg 向量的範數（即它的模長），通常用來衡量模型權重的大小。這是後續調整更新的關鍵數據。
    avg_norm = avg.norm().item()

    if not hasattr(global_model, "history_norms"):
        global_model.history_norms = []
    global_model.history_norms.append(avg_norm)
    if len(global_model.history_norms) > 15:
        global_model.history_norms.pop(0)

    min_avg_norm = min(global_model.history_norms) if len(global_model.history_norms) > 1 else 10.0
    max_avg_norm = max(global_model.history_norms) if len(global_model.history_norms) > 1 else 50.0

   
    dynamic_range = max_avg_norm - min_avg_norm
    adjusted_max_avg_norm = min_avg_norm + dynamic_range * min(1.0, current_round / 100.0)

    early_stage_momentum = (
        values.size(0) < min_clients_threshold and
        current_round < momentum_exit_round and
        local_epoch >= 4
    )

    if values.size(0) < min_clients_threshold and not early_stage_momentum:
        print(f"[INFO] round={current_round} class={class_id} | client 太少（{values.size(0)}），跳過 momentum")
        return avg.detach().clone()
    
    if avg_norm < 1e-3 or math.isnan(avg_norm) or math.isinf(avg_norm):
        print(f"[WARNING] round={current_round} class={class_id} | avg 無效，fallback to old_wb")
        return old_wb.detach().clone()
 
    delta = avg - old_wb
    raw_ratio = (num_clients - 2) / 18
    client_scale_ratio = min(0.778, max(0.0, raw_ratio))
    scale_factor = 0.1 + 0.6 * client_scale_ratio
    base_beta = 0.95 - 0.15 * client_scale_ratio
    delta_clip = 2.0 + 6.0 * client_scale_ratio
    max_mom_norm = 10.0 + 40.0 * client_scale_ratio
    print(f"[Dynamic Params - Round {current_round}]")
    print(f"  client_scale_ratio: {client_scale_ratio:.4f}")
    print(f"  scale_factor      : {scale_factor:.4f}")
    print(f"  base_beta         : {base_beta:.4f}")
    print(f"  delta_clip        : {delta_clip:.4f}")
    print(f"  max_mom_norm      : {max_mom_norm:.4f}")
    
    # wandb.log({
    #     "client_scale_ratio": client_scale_ratio,
    #     "scale_factor": scale_factor,
    #     "base_beta": base_beta,
    #     "delta_clip": delta_clip,
    #     "max_mom_norm": max_mom_norm,
    # }, step=current_round)

    if early_stage_momentum:
        momentum_beta = max(0.5, min(0.95, base_beta * (1.0 - current_round / (momentum_exit_round + 1e-6))))

    else:
        momentum_beta = min(0.95, base_beta)

    local_epoch_scale = max(0.6, 1.0 / (1.0 + 0.2 * (local_epoch - 1)))
    delta_clip *= local_epoch_scale
    scale_factor *= local_epoch_scale

    delta_norm_val = delta.norm()
    if delta_norm_val > delta_clip:
        delta = delta * (delta_clip / (delta_norm_val + 1e-6))
    delta = delta * scale_factor

    if not hasattr(global_model, "logits_momentum"):
        global_model.logits_momentum = {}
    if class_id not in global_model.logits_momentum:
        global_model.logits_momentum[class_id] = torch.zeros_like(delta)

    prev_momentum = global_model.logits_momentum[class_id]
    new_momentum = momentum_beta * prev_momentum + (1 - momentum_beta) * delta

    mom_norm = new_momentum.norm()
    if torch.any(torch.isnan(new_momentum)) or torch.any(torch.isinf(new_momentum)):
        print(f"[WARNING] round={current_round} class={class_id} | momentum 出錯，使用 prev")
        new_momentum = prev_momentum.detach().clone()
    elif mom_norm > max_mom_norm:
        print(f"[CLIP] round={current_round} class={class_id} | momentum norm ({mom_norm:.4f}) > {max_mom_norm}, 進行 clip")
        new_momentum = new_momentum * (max_mom_norm / (mom_norm + 1e-6))

    global_model.logits_momentum[class_id] = new_momentum

    avg_norm_ratio = min(1.0, adjusted_max_avg_norm / (avg_norm + 1e-6))
    alpha_base = 1 / (1 + math.exp(-0.03 * (current_round - 60))) * 0.8
    alpha = alpha_base * avg_norm_ratio

    updated = (1 - alpha) * avg + alpha * (old_wb + new_momentum)

    if current_round < 40:
        if not hasattr(global_model, "safe_copy"):
            global_model.safe_copy = {}
        global_model.safe_copy[class_id] = avg.detach().clone()

    if not hasattr(global_model, "global_momentum"):
        global_model.global_momentum = {}
    global_model.global_momentum[weight_key] = new_momentum[:-1].detach().clone()
    global_model.global_momentum[bias_key] = new_momentum[-1].detach().clone()

    dist_old_to_avg = (avg - old_wb).norm()
    dist_old_to_updated = (updated - old_wb).norm()
    ratio_thresh = 2.5 if current_round >= 100 else 2.0
    if dist_old_to_updated > dist_old_to_avg * ratio_thresh:
        print(f"[WARNING] round={current_round} class={class_id} | updated 偏離過大，使用 avg")
        updated = avg.detach().clone()

    if updated.norm().item() < 1e-5 or torch.isnan(updated).any() or torch.isinf(updated).any() or updated.norm().item() > 300.0:
        print(f"[FALLBACK] round={current_round} class={class_id} | updated 無效，回退 safe copy")
        updated = global_model.safe_copy.get(class_id, avg.detach().clone())

    if not hasattr(global_model, "rolling_stats"):
        global_model.rolling_stats = {
            "acc_history": [],
            "last_updated_class_params": {},
            "last_valid_round": -1,
            "panic_mode": False
        }
    
   
    global_model.rolling_stats["last_updated_class_params"][class_id] = old_wb.detach().clone()
    global_model.rolling_stats["last_valid_round"] = current_round

    if global_model.rolling_stats.get("panic_mode", False):
        print(f"[PANIC] round={current_round} class={class_id} | 準確率崩潰，回復上次參數")
        return global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())

    print(
        f"[DEBUG] round={current_round:3d} | clients={values.size(0)} | "
        f"∆norm={delta.norm():.4f} | mom_norm={mom_norm:.4f} | "
        f"new_mom.norm={new_momentum.norm():.4f} | "
        f"updated.norm={updated.norm():.4f} | updated.mean={updated.mean().item():.4f} | "
        f"α={alpha:.2f} | base_β={base_beta:.2f} | "
        f"scale={scale_factor:.3f} | epoch_scale={local_epoch_scale:.3f} | "
        f"∆clip={delta_clip:.2f} | avg.norm={avg_norm:.4f} | avg.mean={avg.mean().item():.4f}"
    )
    if hasattr(global_model, "rolling_stats"):
        acc_hist = global_model.rolling_stats.get("acc_history", [])
        if len(acc_hist) >= 6:
            recent_avg = np.mean(acc_hist[-3:])
            prev_avg = np.mean(acc_hist[-6:-3])
            if prev_avg > 0.3 and (recent_avg < 0.5 * prev_avg):
                print(f"[PANIC MODE] round={current_round} class={class_id} | Accuracy drop detected: {prev_avg:.4f} -> {recent_avg:.4f}")
                global_model.rolling_stats["panic_mode"] = True
                global_model.rolling_stats["panic_round"] = current_round
                global_model.rolling_stats["panic_class_id"] = class_id
                updated = global_model.rolling_stats["last_updated_class_params"].get(class_id, avg.detach().clone())
                return updated

    return updated