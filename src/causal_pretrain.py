import logging
import math

import numpy as np
import torch
import torch.nn.functional as F


def build_causal_matrices(data_pretrain_flat, voc_size,
                           method='ipw_cpmi', eps=1e-6,
                           min_co_count=1, smoothing=0.5):

    n_d, n_p, n_m = voc_size[0], voc_size[1], voc_size[2]
    n_visits = len(data_pretrain_flat)

    # === Step 1: 构建共现频次表（向量化） ===
    # diag→med 共现矩阵
    co_dm = np.zeros((n_d, n_m), dtype=np.float64)
    co_pm = np.zeros((n_p, n_m), dtype=np.float64)
    freq_d = np.zeros(n_d, dtype=np.float64)
    freq_p = np.zeros(n_p, dtype=np.float64)
    freq_m = np.zeros(n_m, dtype=np.float64)

    for visit in data_pretrain_flat:
        diag_ids = visit[0]
        proc_ids = visit[1]
        med_ids = visit[2] if len(visit) > 2 else []

        for d in diag_ids:
            freq_d[d] += 1.0
        for p in proc_ids:
            freq_p[p] += 1.0
        for m in med_ids:
            freq_m[m] += 1.0
            for d in diag_ids:
                co_dm[d, m] += 1.0
            for p in proc_ids:
                co_pm[p, m] += 1.0

    # === Step 2: 计算 cPMI 矩阵 ===
    # P(m_j) 全局边际概率
    p_m = freq_m / max(n_visits, 1)  # [|M|]

    if method == 'cpmi':
        # 标准 cPMI: log(P(m|d) / P(m))
        # P(m|d) = co_dm[d, m] / freq_d[d]
        # 等价于: log(co_dm[d,m] * N / (freq_d[d] * freq_m[m]))
        with np.errstate(divide='ignore', invalid='ignore'):
            denom_d = freq_d[:, None] * freq_m[None, :]  # [|D|, |M|]
            M_dm_signed = np.log((co_dm * n_visits + eps) / (denom_d + eps))
            denom_p = freq_p[:, None] * freq_m[None, :]
            M_pm_signed = np.log((co_pm * n_visits + eps) / (denom_p + eps))

    elif method == 'ipw_cpmi':
        # IPW 修正: log(P(m|d) / P(m|¬d))
        # P(m|d)  = co_dm[d, m] / freq_d[d]
        # P(m|¬d) = (freq_m[m] - co_dm[d, m]) / (N - freq_d[d])
        # 加性平滑避免 0
        with np.errstate(divide='ignore', invalid='ignore'):
            # diag-med
            num_d = (co_dm + smoothing) / (freq_d[:, None] + 2 * smoothing + eps)
            comp_d = freq_m[None, :] - co_dm  # 没有得 d 但得 m 的次数
            comp_n_d = n_visits - freq_d[:, None]  # 没有得 d 的总次数
            den_d = (comp_d + smoothing) / (comp_n_d + 2 * smoothing + eps)
            M_dm_signed = np.log((num_d + eps) / (den_d + eps))

            # proc-med
            num_p = (co_pm + smoothing) / (freq_p[:, None] + 2 * smoothing + eps)
            comp_p = freq_m[None, :] - co_pm
            comp_n_p = n_visits - freq_p[:, None]
            den_p = (comp_p + smoothing) / (comp_n_p + 2 * smoothing + eps)
            M_pm_signed = np.log((num_p + eps) / (den_p + eps))
    else:
        raise ValueError(f"Unknown method: {method}")

    # === Step 3: 后处理 — NaN/Inf 清理 + 共现阈值过滤 ===
    M_dm_signed = np.nan_to_num(M_dm_signed, nan=0.0, posinf=10.0, neginf=-10.0)
    M_pm_signed = np.nan_to_num(M_pm_signed, nan=0.0, posinf=10.0, neginf=-10.0)
    M_dm_signed = np.clip(M_dm_signed, -10.0, 10.0)
    M_pm_signed = np.clip(M_pm_signed, -10.0, 10.0)

    # 共现次数过低的清零（去除偶然关联）
    M_dm_signed[co_dm < min_co_count] = 0.0
    M_pm_signed[co_pm < min_co_count] = 0.0

    # === Step 4: 非负版本（用于 CER 软目标）===
    M_dm_pos = np.clip(M_dm_signed, 0.0, None)
    M_pm_pos = np.clip(M_pm_signed, 0.0, None)

    # === 统计信息 ===
    stats = {
        'n_visits': n_visits,
        'method': method,
        'M_dm_nonzero_ratio': float((M_dm_pos > 0).mean()),
        'M_pm_nonzero_ratio': float((M_pm_pos > 0).mean()),
        'M_dm_max': float(M_dm_pos.max()),
        'M_pm_max': float(M_pm_pos.max()),
        'M_dm_signed_min': float(M_dm_signed.min()),
        'M_dm_signed_max': float(M_dm_signed.max()),
    }

    logging.info("=" * 70)
    logging.info("[CER] 因果效应矩阵构建完成")
    logging.info(f"  方法: {method}")
    logging.info(f"  Visit 数: {n_visits}, |D|={n_d}, |P|={n_p}, |M|={n_m}")
    logging.info(f"  M_dm 非零比例: {stats['M_dm_nonzero_ratio']*100:.2f}%, "
                 f"max={stats['M_dm_max']:.4f}")
    logging.info(f"  M_pm 非零比例: {stats['M_pm_nonzero_ratio']*100:.2f}%, "
                 f"max={stats['M_pm_max']:.4f}")
    logging.info(f"  signed 矩阵范围: [{stats['M_dm_signed_min']:.4f}, "
                 f"{stats['M_dm_signed_max']:.4f}]")
    logging.info("=" * 70)

    return (M_dm_pos.astype(np.float32),
            M_dm_signed.astype(np.float32),
            M_pm_pos.astype(np.float32),
            M_pm_signed.astype(np.float32),
            stats)


# ============================================================================
# 2. CER 训练目标构造（在线，每 batch 计算）
# ============================================================================

def compute_cer_target_distribution(visit, M_dm_pos, M_pm_pos,
                                      relevance_decay=0.1,
                                      temperature=1.0,
                                      device=None):
    """
    为单个 visit 构造 CER 软目标分布。

    u_visit[j] = softmax( (sum_i M_dm[d_i, j] * w_i + sum_k M_pm[p_k, j] * w_k) / T )

    其中:
        w_i = exp(-α * i)  — 按 priority 位置衰减的权重（CERaMoE 数据按
                              relevance 排序，position 0 最重要）
        T   = temperature  — 温度系数，越小目标越尖锐

    Args:
        visit: [diag_ids, proc_ids, med_ids, weight]
        M_dm_pos: [|D|, |M|] tensor，非负 cPMI
        M_pm_pos: [|P|, |M|] tensor
        relevance_decay: α
        temperature: T
        device: 输出张量的设备

    Returns:
        target: [|M|] tensor — 概率分布（sum = 1）
        raw_score: [|M|] tensor — 加权前的累积因果分数（用于诊断）
    """
    diag_ids = visit[0]
    proc_ids = visit[1]

    if device is None:
        device = M_dm_pos.device

    n_m = M_dm_pos.shape[1]
    raw_score = torch.zeros(n_m, device=device, dtype=M_dm_pos.dtype)

    # 诊断累积（带 priority 衰减）
    if len(diag_ids) > 0:
        weights_d = torch.tensor(
            [math.exp(-relevance_decay * i) for i in range(len(diag_ids))],
            device=device, dtype=M_dm_pos.dtype
        ).unsqueeze(1)  # [n_d, 1]
        diag_idx = torch.LongTensor(diag_ids).to(device)
        raw_score = raw_score + (M_dm_pos[diag_idx] * weights_d).sum(0)

    # 操作累积（带 priority 衰减）
    if len(proc_ids) > 0:
        weights_p = torch.tensor(
            [math.exp(-relevance_decay * i) for i in range(len(proc_ids))],
            device=device, dtype=M_pm_pos.dtype
        ).unsqueeze(1)
        proc_idx = torch.LongTensor(proc_ids).to(device)
        raw_score = raw_score + (M_pm_pos[proc_idx] * weights_p).sum(0)

    # 转化为软目标分布
    if raw_score.sum() < 1e-9:
        # 当前 visit 在因果矩阵中找不到有效信号 — 使用均匀分布
        target = torch.ones(n_m, device=device, dtype=raw_score.dtype) / n_m
    else:
        target = F.softmax(raw_score / max(temperature, 1e-6), dim=0)

    return target, raw_score


def cer_loss(pred_logits, target_distributions, temperature=1.0):
    """
    CER 损失：KL 散度 KL(target || softmax(pred_logits / T))。

    Args:
        pred_logits: [batch, |M|] — 模型输出（cls_cer 头）
        target_distributions: [batch, |M|] — 软目标
        temperature: 模型预测分布的温度

    Returns:
        loss: scalar
    """
    pred_log_probs = F.log_softmax(pred_logits / max(temperature, 1e-6), dim=-1)
    # KL(target || pred) = sum target * (log target - log pred)
    # PyTorch 的 kl_div(input=log_p, target=q) 计算 KL(q || p) = sum q * (log q - log p)
    loss = F.kl_div(pred_log_probs, target_distributions,
                     reduction='batchmean', log_target=False)
    return loss


# ============================================================================
# 3. 因果显著度（用于 Rarity Router 增强）
# ============================================================================

def compute_visit_causal_salience(visit, M_dm_pos, M_pm_pos, eps=1e-6):
    """
    计算 visit 的"因果信号峰度" — 衡量该 visit 的 diag/proc 组合
    是否强烈指向某些特定药物。

    salience ∈ [0, 1]:
      - 接近 0: 因果信号扁平/缺失（罕见或不寻常的组合）
      - 接近 1: 因果信号尖锐（少数几个药物被强烈指示）

    Args:
        visit: [diag_ids, proc_ids, ...]
        M_dm_pos, M_pm_pos: 非负 cPMI 矩阵 (numpy 或 tensor)

    Returns:
        salience: float (peakedness, 即 max / sum)
        magnitude: float (信号强度, log(1 + max))
    """
    if isinstance(M_dm_pos, torch.Tensor):
        M_dm_pos = M_dm_pos.detach().cpu().numpy()
        M_pm_pos = M_pm_pos.detach().cpu().numpy()

    diag_ids = visit[0]
    proc_ids = visit[1]
    n_m = M_dm_pos.shape[1]

    raw = np.zeros(n_m, dtype=np.float32)
    if len(diag_ids) > 0:
        raw += M_dm_pos[diag_ids].sum(axis=0)
    if len(proc_ids) > 0:
        raw += M_pm_pos[proc_ids].sum(axis=0)

    total = raw.sum() + eps
    peak = raw.max()
    salience = float(peak / total)
    magnitude = float(math.log(1.0 + peak))
    return salience, magnitude


# ============================================================================
# 4. 因果偏置校准（fine-tune 阶段）
# ============================================================================

def compute_causal_bias(visit, M_dm_signed, M_pm_signed,
                         standardize=True, eps=1e-6):
    """
    为 visit 计算每个药物的因果偏置项（带正负号），用于 fine-tune 阶段
    对模型 logits 进行校准。

    bias[j] = sum_{d_k ∈ diag} M_dm_signed[d_k, j]
            + sum_{p_k ∈ proc} M_pm_signed[p_k, j]

    若 standardize=True，则对 bias 做 z-score 标准化（按 visit 内部）。

    Args:
        visit: [diag_ids, proc_ids, ...]
        M_dm_signed, M_pm_signed: tensor [|D|/|P|, |M|]，含正负值
        standardize: 是否做 visit 内 z-score
        eps: 标准化数值稳定项

    Returns:
        bias: [|M|] tensor
    """
    diag_ids = visit[0]
    proc_ids = visit[1]
    device = M_dm_signed.device
    n_m = M_dm_signed.shape[1]

    bias = torch.zeros(n_m, device=device, dtype=M_dm_signed.dtype)
    if len(diag_ids) > 0:
        diag_idx = torch.LongTensor(diag_ids).to(device)
        bias = bias + M_dm_signed[diag_idx].sum(0)
    if len(proc_ids) > 0:
        proc_idx = torch.LongTensor(proc_ids).to(device)
        bias = bias + M_pm_signed[proc_idx].sum(0)

    if standardize:
        mu = bias.mean()
        sigma = bias.std() + eps
        bias = (bias - mu) / sigma

    return bias


# ============================================================================
# 5. CER batch 数据准备（与 main 训练循环对接）
# ============================================================================

def prepare_cer_batch_targets(batch_visits, M_dm_pos, M_pm_pos,
                                relevance_decay=0.1, temperature=1.0,
                                device=None):
    """
    一次性为 batch 的所有 visit 计算 CER 软目标。

    Returns:
        target_batch: [batch_size, |M|] tensor
    """
    targets = []
    for visit in batch_visits:
        t, _ = compute_cer_target_distribution(
            visit, M_dm_pos, M_pm_pos,
            relevance_decay=relevance_decay,
            temperature=temperature,
            device=device,
        )
        targets.append(t)
    return torch.stack(targets, dim=0)
