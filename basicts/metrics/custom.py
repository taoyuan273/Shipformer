import torch
import torch.nn.functional as F
import numpy as np
from .mse import masked_mse


PI = 3.141592653589793

debug_log_path = "debug_loss.txt"  # 指定日志文件路径

def log_debug_info(message):
    """ 将 debug 信息写入日志文件 """
    with open(debug_log_path, "a") as debug_log:  # "a" 追加模式
        debug_log.write(message + "\n")

def amp_loss(outputs, targets):  #(7)
    #outputs = B, T, 1 --> B, 1, T
    B,_, T = outputs.shape
    fft_size = 1 << (2 * T - 1).bit_length()
    out_fourier = torch.fft.fft(outputs, fft_size, dim = -1)
    tgt_fourier = torch.fft.fft(targets, fft_size, dim = -1)

    out_norm = torch.norm(outputs, dim = -1, keepdim = True)
    tgt_norm = torch.norm(targets, dim = -1, keepdim = True)

    #calculate normalized auto correlation
    auto_corr = torch.fft.ifft(tgt_fourier * tgt_fourier.conj(), dim = -1).real
    auto_corr = torch.cat([auto_corr[...,-(T-1):], auto_corr[...,:T]], dim = -1)
    nac_tgt = auto_corr / (tgt_norm * tgt_norm)    #自相关

    # calculate cross correlation
    cross_corr = torch.fft.ifft(tgt_fourier * out_fourier.conj(), dim = -1).real
    cross_corr = torch.cat([cross_corr[...,-(T-1):], cross_corr[...,:T]], dim = -1)
    nac_out = cross_corr / (tgt_norm * out_norm)  #互相关
    
    loss = torch.mean(torch.abs(nac_tgt - nac_out))
    return loss


def ashift_loss(outputs, targets):   #(5)
    B, _, T = outputs.shape
    return T * torch.mean(torch.abs(1 / T - torch.softmax(outputs - targets, dim = -1)))


def phase_loss(outputs, targets): #(6)
    B, _, T = outputs.shape
    out_fourier = torch.fft.fft(outputs, dim = -1)
    tgt_fourier = torch.fft.fft(targets, dim = -1)
    tgt_fourier_sq = (tgt_fourier.real ** 2 + tgt_fourier.imag ** 2)
    mask = (tgt_fourier_sq > (T)).float()
    topk_indices = tgt_fourier_sq.topk(k = int(T**0.5), dim = -1).indices
    mask = mask.scatter_(-1, topk_indices, 1.)
    mask[...,0] = 1.
    mask = torch.where(mask > 0, 1., 0.)
    mask = mask.bool()
    not_mask = (~mask).float()
    not_mask /= torch.mean(not_mask)
    out_fourier_sq = (torch.abs(out_fourier.real) + torch.abs(out_fourier.imag))
    zero_error = torch.abs(out_fourier) * not_mask
    zero_error = torch.where(torch.isnan(zero_error), torch.zeros_like(zero_error), zero_error)
    mask = mask.float()
    mask /= torch.mean(mask)
    ae = torch.abs(out_fourier - tgt_fourier) * mask
    ae = torch.where(torch.isnan(ae), torch.zeros_like(ae), ae)
    phase_loss = (torch.mean(zero_error) + torch.mean(ae)) / (T ** .5)
    return phase_loss




def correlation_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算 Pearson 相关性损失 (1 - PCC)，鼓励模型学习目标的方向性和模式。

    参数:
        outputs (torch.Tensor): 模型预测值，形状为 (B, T) 或 (B, C, T)
        targets (torch.Tensor): 真实目标值，形状与 outputs 相同

    返回:
        torch.Tensor: 相关性损失 (标量, 越小越好)
    """
    # 确保输入形状一致
    assert outputs.shape == targets.shape, "Outputs and targets must have the same shape"

    # 计算均值 (沿最后一维进行计算)
    outputs_mean = torch.mean(outputs, dim=-1, keepdim=True)
    targets_mean = torch.mean(targets, dim=-1, keepdim=True)

    # 计算标准差 (unbiased=False 保证计算梯度时数值稳定)
    outputs_std = torch.sqrt(torch.var(outputs, dim=-1, unbiased=False, keepdim=True) + 1e-6)
    targets_std = torch.sqrt(torch.var(targets, dim=-1, unbiased=False, keepdim=True) + 1e-6)

    # 计算协方差 (利用 `torch.mean()` 避免 `sum()` 的归一化问题)
    covariance = torch.mean((outputs - outputs_mean) * (targets - targets_mean), dim=-1, keepdim=True)

    # 计算 Pearson 相关系数 (PCC)
    pcc = covariance / (outputs_std * targets_std + 1e-6)  # 避免除零错误

    # 计算相关性损失 (1 - PCC, 越小越好)
    loss = 1 - pcc

    return loss.mean()  # 返回均值，适用于 mini-batch 训练




def variance_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算基于 KL 散度的方差损失 (Variance Loss)。
    衡量预测值和目标值的局部波动（去均值后）在分布上的匹配程度。

    参数:
        outputs (torch.Tensor): 模型预测值，形状为 (B, T) 或 (B, C, T)
        targets (torch.Tensor): 真实目标值，形状与 `outputs` 相同

    返回:
        torch.Tensor: 方差损失 (越小越好)
    """

    # 确保输入形状一致
    assert outputs.shape == targets.shape, "Outputs and targets must have the same shape"

    # 计算均值 (去均值以获取波动)
    outputs_mean = torch.mean(outputs, dim=-1, keepdim=True)
    targets_mean = torch.mean(targets, dim=-1, keepdim=True)

    # 计算偏差 (去中心化)
    outputs_dev = outputs - outputs_mean
    targets_dev = targets - targets_mean

    # 计算 softmax（转换为概率分布）
    p_out = F.softmax(outputs_dev, dim=-1)  # 预测分布
    p_tgt = F.softmax(targets_dev, dim=-1)  # 目标分布

    # 计算 KL 散度 (P_tgt || P_out)
    kl_div = F.kl_div(p_out.log(), p_tgt, reduction='batchmean')

    return kl_div



def mean_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算预测序列与真实序列在均值层面的 MAE (Mean Absolute Error)，
    公式: L_Mean = 1/N * Σ |μ - μ_hat|

    参数:
        outputs (torch.Tensor): 预测值, 形状 (B, T) 或 (B, C, T)
        targets (torch.Tensor): 真实值, 形状 (B, T) 或 (B, C, T)

    返回:
        torch.Tensor: 均值损失 (标量, 越小越好)
    """

    # 确保输入形状一致
    assert outputs.shape == targets.shape, "Outputs and targets must have the same shape."

    # 计算均值 (batch-wise 均值)
    out_mean = torch.mean(outputs, dim=-1, keepdim=True)
    tgt_mean = torch.mean(targets, dim=-1, keepdim=True)

    # 计算 MAE（绝对误差）
    loss = F.l1_loss(out_mean, tgt_mean, reduction='mean')

    # 计算 MSE（均方误差）
    #loss = F.mse_loss(out_mean, tgt_mean, reduction='mean')

    return loss



def compute_gradient_norm(loss, model_parameters):
    # 使用 autograd.grad 计算梯度，不会影响参数的 .grad 属性
    grads = torch.autograd.grad(loss, model_parameters, create_graph=False, retain_graph=True, allow_unused=True)
    grad_norm_sq = 0.0
    for grad in grads:
        if grad is not None:
            grad_norm_sq += torch.sum(grad ** 2)
    grad_norm = torch.sqrt(grad_norm_sq)
    return grad_norm


def compute_c_v(outputs, targets):
    """
    根据真实值 targets 和预测值 outputs 计算缩放因子 c 和 v.
    参数:
        outputs: 模型预测值, 形状 (B, T) 或 (B, C, T)
        targets: 真实目标值, 形状与 outputs 相同
    返回:
        c, v: 均为标量, 取值范围 [0, 1]
    公式:
        c = 1/2 * (1 + cov(outputs, targets) / (std(outputs) * std(targets)))
        v = 2 * std(outputs) * std(targets) / (std(outputs)^2 + std(targets)^2)
    """
    # 将输出展平，便于统计计算
    outputs_flat = outputs.reshape(-1)
    targets_flat = targets.reshape(-1)

    mean_out = torch.mean(outputs_flat)
    mean_tgt = torch.mean(targets_flat)

    std_out = torch.std(outputs_flat, unbiased=False)
    std_tgt = torch.std(targets_flat, unbiased=False)

    # 协方差计算
    cov = torch.mean((outputs_flat - mean_out) * (targets_flat - mean_tgt))

    eps = 1e-8  # 防止除零
    c = 0.5 * (1.0 + (cov / (std_out * std_tgt + eps)))
    v = (2.0 * std_out * std_tgt) / (std_out ** 2 + std_tgt ** 2 + eps)

    # 限制在 [0, 1] 范围内
    c = torch.clamp(c, 0.0, 1.0)
    v = torch.clamp(v, 0.0, 1.0)

    return c, v


def compute_loss_weights(corr_loss, var_loss, mean_loss, model_parameters, outputs, targets):
    """
    根据三个损失函数的梯度大小计算动态权重
    参数:
        corr_loss: 相关性损失
        var_loss: 方差损失
        mean_loss: 均值损失
        model_parameters: 模型输出层或整个模型的参数列表
        c, v: 均值损失的缩放因子（根据真实值与预测值的协方差与标准差计算得到）
    返回:
        alpha, beta, gamma 分别为三个损失的权重
    """
    # 使用 autograd.grad 计算各个损失的梯度L2范数（不会改变参数的 .grad 属性）
    G_corr = compute_gradient_norm(corr_loss, model_parameters)
    G_var = compute_gradient_norm(var_loss, model_parameters)
    G_mean = compute_gradient_norm(mean_loss, model_parameters)

    # 计算平均梯度
    G_avg = (G_corr + G_var + G_mean) / 3.0

    # 计算缩放因子 c 和 v（基于当前 batch 的 outputs 与 targets）
    c, v = compute_c_v(outputs, targets)

    # 根据公式计算权重：防止除零加上 1e-8
    alpha = G_avg / (G_corr + 1e-8)  # 对相关性损失
    beta = G_avg / (G_var + 1e-8)  # 对方差损失
    gamma = (c * v) / (G_mean + 1e-8)  # 对均值损失
    
        # 调试信息
    #log_debug_info(f"[DEBUG] Epoch {epoch}, Batch {iter_index}")
    #print(f"[DEBUG] G_corr: {G_corr.item()}, G_var: {G_var.item()}, G_mean: {G_mean.item()}")
    #log_debug_info(f"[DEBUG] G_corr: {G_corr.item()}, G_var: {G_var.item()}, G_mean: {G_mean.item()}")
    #print(f"[DEBUG] c: {c.item()}, v: {v.item()}, alpha: {alpha.item()}, beta: {beta.item()}, gamma: {gamma.item()}")
    #log_debug_info(f"[DEBUG] c: {c.item()}, v: {v.item()}, alpha: {alpha.item()}, beta: {beta.item()}, gamma: {gamma.item()}")

    return alpha, beta, gamma



def composite_loss(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan, base_alpha=0,
                   base_gamma=1.0, extra_weight=0, extra_weight_1=2,model=None) -> torch.Tensor:
    """
    计算综合损失：
      1. Tilde-Q 损失部分（loss_base）
      2. 额外损失部分（loss_extra）
      3. 使用 `null_val` 进行掩码，确保只计算有效值的损失
      4. 自动适配 GPU/CPU 设备

    参数:
        prediction (torch.Tensor): 预测值，形状为 (B, C, T) 或 (B, T, C)
        target (torch.Tensor): 真实值，与 `prediction` 形状相同
        null_val (float): 用于掩码的无效值，默认 `np.nan`
        base_alpha (float): 控制 `l_ashift` 和 `l_phase` 的权重，默认 0.5
        base_gamma (float): 控制 `l_amp` 的权重，默认 0.0
        extra_weight (float): 控制额外损失部分的整体权重，默认 1.0
        model (torch.nn.Module, optional): 训练模型实例，用于计算动态权重

    返回:
        torch.Tensor: 标量损失值
    """

    # 确保传入的 `model` 不是 None
    if model is None:
        raise ValueError("需要传入模型实例以提取模型参数, 如 model=your_model")

    # 设备自动适配（CUDA/GPU 或 CPU）
    device = target.device
    model_parameters = list(model.parameters())
        # 计算 MAE
    mse_loss = masked_mse(prediction, target, null_val=null_val)

    # 处理掩码（Masked Loss），忽略 `null_val` 位置
    eps = 5e-5
    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        mask = ~torch.isclose(target, torch.tensor(null_val, device=device).expand_as(target), atol=eps, rtol=0.0)

    mask = mask.float()
    prediction, target = prediction * mask, target * mask
    prediction = torch.nan_to_num(prediction)
    target = torch.nan_to_num(target)
   
    # Debug 打印，查看维度
    #print(f"[DEBUG] Before processing: prediction.shape = {prediction.shape}, target.shape = {target.shape}")

    #处理 4D 数据 `(B, T, C, 1)` → `(B, T, C)`
    if prediction.dim() == 4 and prediction.shape[-1] == 1:  
        prediction = prediction.squeeze(-1)  # (B, T, C, 1) → (B, T, C)
        target = target.squeeze(-1)

    # 统一转换为 `composite_loss()` 需要的 `(B, C, T)`
    prediction = prediction.permute(0, 2, 1)  # (B, T, C) → (B, C, T)
    target = target.permute(0, 2, 1)

    # Debug 打印，查看最终维度
    #print(f"[DEBUG] After processing: prediction.shape = {prediction.shape}, target.shape = {target.shape}")


    # 计算 Tilde-Q 损失部分
    l_ashift = ashift_loss(prediction, target)
    l_amp = amp_loss(prediction, target)
    l_phase = phase_loss(prediction, target)
    #loss_base = base_alpha * l_ashift + (1 - base_alpha) * l_phase + base_gamma * l_amp
    loss_base = base_alpha * l_ashift + 0 * l_phase + base_gamma * l_amp

    # 计算额外损失部分
    corr_loss_val = correlation_loss(prediction, target)
    var_loss_val = variance_loss(prediction, target)
    mean_loss_val = mean_loss(prediction, target)
    #print(f"[DEBUG] corr_loss_val: {corr_loss_val.item():.6f}, var_loss_val: {var_loss_val.item():.6f}, mean_loss_val: {mean_loss_val.item():.6f}")
    #log_debug_info(f"[DEBUG] corr_loss_val: {corr_loss_val.item():.6f}, var_loss_val: {var_loss_val.item():.6f}, mean_loss_val: {mean_loss_val.item():.6f}")
    # 计算动态权重
    #w_corr, w_var, w_mean = compute_loss_weights(corr_loss_val, var_loss_val, mean_loss_val, model_parameters,prediction, target)
    #print(f"[DEBUG] w_corr: {w_corr.item():.6f}, w_var: {w_var.item():.6f}, w_mean: {w_mean.item():.6f}")
    #log_debug_info(f"[DEBUG] w_corr: {w_corr.item():.6f}, w_var: {w_var.item():.6f}, w_mean: {w_mean.item():.6f}")
    
    loss_extra =  0 * corr_loss_val +  0 * var_loss_val + 1* mean_loss_val

    # 计算总损失
    total_loss = loss_base + extra_weight * loss_extra + extra_weight_1 * mse_loss
    total_loss = torch.nan_to_num(total_loss)  # 避免 `NaN` 影响梯度

    return total_loss  # 确保返回值在正确的设备上

