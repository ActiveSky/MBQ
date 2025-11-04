import torch

@torch.no_grad()
def pseudo_quantize_tensor(tensor, n_bits=8, zero_point=True, q_group_size=-1, per_tensor=False, inplace=False):
    """
    对张量进行伪量化，支持权重、激活和 KV 缓存的量化。

    Args:
        tensor (torch.Tensor): 要量化的输入张量。
        n_bits (int, optional): 量化的位宽。默认为 8。
        zero_point (bool, optional): 是否使用零点量化。默认为 True。
        q_group_size (int, optional): 量化组的大小。如果大于 0，则按组进行量化。默认为 -1。
        per_tensor (bool, optional): 是否按张量进行量化（即整个张量使用一个比例因子和零点）。默认为 False。
        inplace (bool, optional): 是否原地修改张量。默认为 False。

    Returns:
        torch.Tensor: 伪量化后的张量。
    """
    org_tensor_shape = tensor.shape # 记录原始张量形状
    if q_group_size > 0:
        # 如果指定了量化组大小，则重塑张量以按组进行处理
        assert org_tensor_shape[-1] % q_group_size == 0, "张量的最后一个维度必须是量化组大小的倍数。"
        tensor = tensor.reshape(-1, q_group_size)
    if per_tensor:
        # 如果是按张量量化，则将张量重塑为一维
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2, "张量必须是二维的，以便进行量化计算。" # 确保张量是二维的

    if zero_point:
        # 使用零点量化
        max_val = tensor.amax(dim=1, keepdim=True) # 计算每行的最大值
        min_val = tensor.amin(dim=1, keepdim=True) # 计算每行的最小值
        max_int = 2**n_bits - 1 # 量化后的最大整数值
        min_int = 0 # 量化后的最小整数值
        scales = (max_val - min_val).clamp(min=1e-5) / max_int # 计算比例因子，防止除以零
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int) # 计算零点
    else:
        # 不使用零点量化（对称量化）
        max_val = tensor.abs().amax(dim=1, keepdim=True) # 计算每行的绝对值最大值
        max_val = max_val.clamp(min=1e-5) # 防止除以零
        max_int = 2 ** (n_bits - 1) - 1 # 量化后的最大整数值
        min_int = -(2 ** (n_bits - 1)) # 量化后的最小整数值
        scales = max_val / max_int # 计算比例因子
        zeros = 0 # 零点为 0

    if inplace:
        # 原地量化
        (
            (tensor.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        # 非原地量化
        tensor = (
            torch.clamp(torch.round(tensor / scales) + zeros, min_int, max_int) - zeros
        ) * scales

    assert torch.isnan(tensor).sum() == 0, "量化后的张量中包含 NaN 值。" # 检查量化后张量是否包含 NaN

    tensor = tensor.reshape(org_tensor_shape) # 将张量重塑回原始形状

    # 返回量化后的张量，以及可选的比例因子和零点值
    # return tensor, scales.view(tensor.shape[0], -1), zeros.view(tensor.shape[0], -1)
    return tensor


@torch.no_grad()
def quantize_weight_per_channel_absmax(w, n_bits=8, zero_point=False):
    """
    使用每通道绝对值最大值方法对权重进行量化。
    该函数通过调用 `pseudo_quantize_tensor` 实现，其中 `per_tensor` 设置为 False，表示按通道量化。

    Args:
        w (torch.Tensor): 要量化的权重张量。
        n_bits (int, optional): 量化的位宽。默认为 8。
        zero_point (bool, optional): 是否使用零点量化。默认为 False。

    Returns:
        torch.Tensor: 量化后的权重张量。
    """
    # 调用伪量化函数，按通道进行量化 (per_tensor=False)
    tensor = pseudo_quantize_tensor(w, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=False, inplace=False)
    return tensor
    
@torch.no_grad()
def quantize_activation_per_token_absmax(t, n_bits=8, zero_point=False):
    """
    使用每 token 绝对值最大值方法对激活进行量化。
    该函数将输入张量重塑为二维，然后调用 `pseudo_quantize_tensor` 进行量化，
    其中 `per_tensor` 设置为 False，表示按 token（即每行）量化。

    Args:
        t (torch.Tensor): 要量化的激活张量。
        n_bits (int, optional): 量化的位宽。默认为 8。
        zero_point (bool, optional): 是否使用零点量化。默认为 False。

    Returns:
        torch.Tensor: 量化后的激活张量，形状与输入相同。
    """
    t_shape = t.shape # 记录原始张量形状
    t = t.view(-1, t_shape[-1]) # 将张量重塑为二维，每行代表一个 token 的激活
    # 调用伪量化函数，按 token 进行量化 (per_tensor=False)
    t = pseudo_quantize_tensor(t, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=False, inplace=False)
    return t.reshape(t_shape) # 将量化后的张量重塑回原始形状
    
@torch.no_grad()
def quantize_weight_per_tensor_absmax(w, n_bits=8, zero_point=False):
    """
    使用每张量绝对值最大值方法对权重进行量化。
    该函数通过调用 `pseudo_quantize_tensor` 实现，其中 `per_tensor` 设置为 True，表示整个张量使用一个比例因子和零点。

    Args:
        w (torch.Tensor): 要量化的权重张量。
        n_bits (int, optional): 量化的位宽。默认为 8。
        zero_point (bool, optional): 是否使用零点量化。默认为 False。

    Returns:
        torch.Tensor: 量化后的权重张量。
    """
    # 调用伪量化函数，按张量进行量化 (per_tensor=True)
    tensor = pseudo_quantize_tensor(w, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=True, inplace=False)
    return tensor
    
@torch.no_grad()
def quantize_activation_per_tensor_absmax(t, n_bits=8, zero_point=False):
    """
    使用每张量绝对值最大值方法对激活进行量化。
    该函数将输入张量重塑为二维，然后调用 `pseudo_quantize_tensor` 进行量化，
    其中 `per_tensor` 设置为 True，表示整个张量使用一个比例因子和零点。

    Args:
        t (torch.Tensor): 要量化的激活张量。
        n_bits (int, optional): 量化的位宽。默认为 8。
        zero_point (bool, optional): 是否使用零点量化。默认为 False。

    Returns:
        torch.Tensor: 量化后的激活张量，形状与输入相同。
    """
    t_shape = t.shape # 记录原始张量形状
    t = t.view(-1, t_shape[-1]) # 将张量重塑为二维，每行代表一个 token 的激活
    # 调用伪量化函数，按张量进行量化 (per_tensor=True)
    t = pseudo_quantize_tensor(t, n_bits=n_bits, zero_point=zero_point, q_group_size=-1, per_tensor=True, inplace=False)
    return t.reshape(t_shape) # 将量化后的张量重塑回原始形状
