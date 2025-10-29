import gc
import torch
import torch.nn as nn

from transformers.models.bloom.modeling_bloom import BloomBlock, BloomGelu
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm
from transformers.activations import GELUActivation

from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

from .qmodule import ScaledActivation
from qmllm.utils.search import get_op_by_name, get_op_name, set_op_by_name
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor

__all__ = ["auto_scale_block", "apply_scale"]


@torch.no_grad()
def get_weight_scale(weight, q_group_size=-1):
    """
    计算权重的量化尺度。

    Args:
        weight (torch.Tensor): 模型的权重张量。
        q_group_size (int, optional): 量化组的大小。如果为-1，则不进行分组。默认为-1。

    Returns:
        torch.Tensor: 计算出的权重尺度。
    """
    org_shape = weight.shape # 记录原始形状
    if q_group_size > 0: # 如果指定了量化组大小
        weight = weight.view(-1, q_group_size) # 将权重重塑为 (num_groups, q_group_size)
    # 计算每个组内权重的绝对值最大值作为尺度
    scale = weight.abs() / weight.abs().amax(dim=1, keepdim=True)
    scale = scale.view(org_shape) # 将尺度重塑回原始形状
    scale = scale.mean(0) # 对尺度进行平均
    return scale


@torch.no_grad()
def get_act_scale(x):
    """
    计算激活的量化尺度。

    Args:
        x (torch.Tensor): 激活张量。

    Returns:
        torch.Tensor: 计算出的激活尺度。
    """
    # 计算激活张量绝对值的平均值作为尺度
    return x.abs().view(-1, x.shape[-1]).mean(0)


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales):
    """
    对LayerNorm层和后续的全连接层进行尺度调整。

    Args:
        ln (nn.LayerNorm or LlamaRMSNorm): LayerNorm层。
        fcs (list or nn.Linear): 后续的全连接层（可以是单个或列表）。
        scales (torch.Tensor): 用于调整的尺度张量。
    """
    if not isinstance(fcs, list): # 如果fcs不是列表，则转换为列表
        fcs = [fcs]

    scales = scales.to(ln.weight.device) # 将尺度移到LayerNorm层的设备

    ln.weight.div_(scales) # LayerNorm的权重除以尺度
    if hasattr(ln, "bias") and ln.bias is not None: # 如果LayerNorm有偏置，则偏置也除以尺度
        ln.bias.div_(scales)

    for fc in fcs: # 遍历所有全连接层
        fc.weight.mul_(scales.view(1, -1)) # 全连接层的权重乘以尺度

    for p in ln.parameters(): # 检查LayerNorm参数中是否有NaN
        assert torch.isnan(p).sum() == 0
    for fc in fcs: # 检查全连接层参数中是否有NaN
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales):
    """
    对两个连续的全连接层进行尺度调整。

    Args:
        fc1 (nn.Linear): 第一个全连接层。
        fc2 (nn.Linear): 第二个全连接层。
        scales (torch.Tensor): 用于调整的尺度张量。
    """
    assert isinstance(fc1, nn.Linear) # 确保fc1是nn.Linear类型
    assert isinstance(fc2, nn.Linear) # 确保fc2是nn.Linear类型
    # assert fc1.out_features == fc2.in_features # 确保fc1的输出特征数与fc2的输入特征数匹配

    scales = scales.to(fc1.weight.device) # 将尺度移到fc1权重的设备

    # fc1.weight.div_(scales.view(-1, 1)) # 原始注释，可能用于调整fc1的权重
    fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1)) # fc1的权重除以尺度
    if fc1.bias is not None: # 如果fc1有偏置，则偏置也除以尺度
        fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1)) # fc2的权重乘以尺度

    for p in fc1.parameters(): # 检查fc1参数中是否有NaN
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters(): # 检查fc2参数中是否有NaN
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu, fc, scales):
    """
    对GELU激活函数和后续的全连接层进行尺度调整。

    Args:
        gelu (nn.GELU or BloomGelu or GELUActivation): GELU激活函数。
        fc (nn.Linear): 后续的全连接层。
        scales (torch.Tensor): 用于调整的尺度张量。
    """
    assert isinstance(gelu, (nn.GELU, BloomGelu, GELUActivation)) # 确保gelu是GELU类型
    assert isinstance(fc, nn.Linear) # 确保fc是nn.Linear类型

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device)) # 全连接层的权重乘以尺度

    for p in fc.parameters(): # 检查fc参数中是否有NaN
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def auto_scale_block(module, module_kwargs, w_bit, q_config, input_feat, ans_mask, vis_mask, reweight_ratio_dict, loss_mode="mae"):
    """
    对Transformer块进行自动尺度搜索，以优化量化性能。

    此函数通过迭代不同的尺度比例，计算量化前后输出的损失，
    从而找到最佳的尺度因子。

    Args:
        module (nn.Module): 待处理的Transformer块。
        module_kwargs (dict): 模块前向传播所需的关键字参数。
        w_bit (int): 权重的位宽。
        q_config (dict): 量化配置字典。
        input_feat (dict): 包含每个线性层输入特征的字典。
        ans_mask (torch.Tensor): 答案部分的掩码。
        vis_mask (torch.Tensor): 视觉部分的掩码。
        reweight_ratio_dict (dict): 重加权比例字典，包含"attn"和"mlp"的比例。
        loss_mode (str, optional): 损失计算模式，"mse"或"mae"。默认为"mae"。

    Returns:
        list: 包含搜索到的尺度列表，每个元素是一个元组 (prev_op_name, layer_names, scales)。
    """
    # 首先，获取权重伪量化函数
    if w_bit is not None:
        def w_quantize_func(p):
            return pseudo_quantize_tensor(
                p,
                n_bits=w_bit,
                **q_config,
            ).detach()
    else:
        def w_quantize_func(p):
            return p

    # 如果module_kwargs中包含"use_cache"，则移除
    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")

    # 寻找最佳尺度比例的内部函数
    def _search_module_scale(block, linears2scale: list, x, reweight_ratio=None, kwargs={}):
        # w: co, ci (权重形状：输出特征数，输入特征数)
        # x: n, ci (输入特征形状：批次大小，输入特征数)
        x = x.to(next(block.parameters()).device) # 将输入移到块的设备
        with torch.no_grad():
            org_out = block(x, **kwargs) # 获取原始输出
            if isinstance(org_out, tuple): # 如果输出是元组，取第一个元素
                org_out = org_out[0]

        x_max = get_act_scale(x) # 获取激活的尺度

        best_error = float("inf") # 初始化最佳误差为无穷大
        best_ratio = -1 # 初始化最佳比例
        best_scales = None # 初始化最佳尺度

        n_grid = 20 # 搜索网格数量
        history = [] # 记录历史损失

        org_sd = {k: v.cpu() for k, v in block.state_dict().items()} # 保存原始状态字典
        for ratio in range(n_grid): # 遍历不同的尺度比例
            ratio = ratio * 1 / n_grid # 计算当前比例
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1) # 计算尺度
            scales = scales / (scales.max() * scales.min()).sqrt() # 归一化尺度
            for fc in linears2scale: # 对需要缩放的线性层进行操作
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device)) # 权重乘以尺度
                fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1)) # 权重伪量化并除以尺度
            out = block(x, **kwargs) # 获取量化后的输出
            if isinstance(out, tuple): # 如果输出是元组，取第一个元素
                out = out[0]

            # loss = (
            #     (org_out - out).float().pow(2).mean().item()
            # )  # float prevents overflow # 原始的MSE损失计算

            if loss_mode == "mse": # 如果损失模式是MSE
                if ans_mask is not None and vis_mask is not None: # 如果同时有答案掩码和视觉掩码
                    ans_mask_expand = ans_mask.unsqueeze(-1).expand_as(out) # 扩展答案掩码
                    vis_mask_expand = vis_mask.unsqueeze(-1).expand_as(out).cuda() # 扩展视觉掩码并移到CUDA
                    masked_diff_ans = ((org_out - out).float().pow(2) * ans_mask_expand) # 计算答案部分的掩码差异
                    masked_diff_vis = ((org_out - out).float().pow(2) * vis_mask_expand) # 计算视觉部分的掩码差异
                    if reweight_ratio is not None: # 如果有重加权比例
                        loss = masked_diff_ans.sum() / ans_mask_expand.sum() + reweight_ratio * (masked_diff_vis.sum() / vis_mask_expand.sum()) # 计算加权损失
                    else:
                        loss = (
                            (org_out - out).float().pow(2).mean().item() # 否则计算平均MSE损失
                        ) 
                elif ans_mask is not None and vis_mask is None: # 如果只有答案掩码
                    ans_mask_expand = ans_mask.unsqueeze(-1).expand_as(out) # 扩展答案掩码
                    masked_diff = ((org_out - out).float().pow(2) * ans_mask_expand) # 计算答案部分的掩码差异
                    loss = masked_diff.sum() / ans_mask_expand.sum() # 计算平均MSE损失
                else:
                    loss = (
                        (org_out - out).float().pow(2).mean().item() # 否则计算平均MSE损失
                    )  # float prevents overflow
            elif loss_mode == "mae": # 如果损失模式是MAE
                if ans_mask is not None and vis_mask is not None: # 如果同时有答案掩码和视觉掩码
                    ans_mask_expand = ans_mask.unsqueeze(-1).expand_as(out) # 扩展答案掩码
                    vis_mask_expand = vis_mask.unsqueeze(-1).expand_as(out).cuda() # 扩展视觉掩码并移到CUDA
                    masked_diff_ans = ((org_out - out).float().abs() * ans_mask_expand) # 计算答案部分的掩码差异绝对值
                    masked_diff_vis = ((org_out - out).float().abs() * vis_mask_expand) # 计算视觉部分的掩码差异绝对值
                    if reweight_ratio is not None: # 如果有重加权比例
                        loss = (masked_diff_ans.sum() + reweight_ratio * masked_diff_vis.sum()) / (ans_mask_expand.sum() + vis_mask_expand.sum()) # 计算加权损失
                    else:
                        loss = (
                            (org_out - out).float().abs().mean().item() # 否则计算平均MAE损失
                        ) 
                elif ans_mask is not None and vis_mask is None: # 如果只有答案掩码
                    ans_mask_expand = ans_mask.unsqueeze(-1).expand_as(out) # 扩展答案掩码
                    masked_diff = ((org_out - out).float().abs() * ans_mask_expand) # 计算答案部分的掩码差异绝对值
                    loss = masked_diff.sum() / ans_mask_expand.sum() # 计算平均MAE损失
                else:
                    loss = (
                        (org_out - out).float().abs().mean().item() # 否则计算平均MAE损失
                    )  # float prevents overflow

            history.append(loss) # 记录损失
            is_best = loss < best_error # 判断是否是最佳损失
            if is_best:
                best_error = loss # 更新最佳误差
                best_ratio = ratio # 更新最佳比例
                best_scales = scales # 更新最佳尺度
            block.load_state_dict(org_sd) # 恢复块的原始状态
        if best_ratio == -1: # 如果没有找到最佳比例
            print(history) # 打印历史损失
            raise Exception("未找到最佳尺度比例。") # 抛出异常
        # print(best_ratio) # 打印最佳比例
        best_scales = best_scales.view(-1) # 将最佳尺度重塑为一维

        assert torch.isnan(best_scales).sum() == 0, best_scales # 确保最佳尺度中没有NaN
        return best_scales.detach() # 返回最佳尺度

    def _auto_get_scale(prev_op, layers, inp, reweight_ratio=None, module2inspect=None, kwargs={}):
        """
        自动获取量化尺度。

        Args:
            prev_op (nn.Module): 前一个操作模块。
            layers (list): 需要缩放的线性层列表。
            inp (torch.Tensor): 输入特征。
            reweight_ratio (float, optional): 重加权比例。默认为None。
            module2inspect (nn.Module, optional): 如果给定，将检查此模块的输出差异而不是layers。默认为None。
            kwargs (dict, optional): 模块前向传播所需的额外关键字参数。默认为{}。

        Returns:
            tuple: 包含 (prev_op_name, layer_names, scales) 的元组。
        """
        # module2inspect: 如果给定，我们将检查此模块的输出差异而不是layers
        if module2inspect is None:
            assert len(layers) == 1, "如果未指定module2inspect，则layers必须只有一个元素。"
            module2inspect = layers[0]

        scales = _search_module_scale(module2inspect, layers, inp, reweight_ratio, kwargs) # 搜索模块尺度
        scales = scales.detach().cpu() # 将尺度移到CPU并分离
        # prev_op_name, [layer_name], scale
        return (
            get_op_name(module, prev_op), # 前一个操作的名称
            tuple([get_op_name(module, m) for m in layers]), # 需要缩放的层名称列表
            scales, # 尺度
        )

    scales_list = []  # 存储搜索到的尺度列表

    # 根据模块类型，为不同的子模块获取尺度
    if isinstance(module, OPTDecoderLayer):
        # 注意力输入
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn_layer_norm, # 前一个操作是自注意力层归一化
                layers=[
                    module.self_attn.q_proj, # Q投影层
                    module.self_attn.k_proj, # K投影层
                    module.self_attn.v_proj, # V投影层
                ],
                inp=input_feat["self_attn.q_proj"], # 输入特征
                module2inspect=module.self_attn, # 检查自注意力模块的输出差异
                kwargs=module_kwargs, # 关键字参数
            )
        )
        # 注意力输出
        scales_list.append(
            _auto_get_scale(
                prev_op=module.self_attn.v_proj, # 前一个操作是V投影层
                layers=[module.self_attn.out_proj], # 输出投影层
                inp=input_feat["self_attn.out_proj"], # 输入特征
            )
        )
        # fc1 (第一个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.final_layer_norm, # 前一个操作是最终层归一化
                layers=[module.fc1], # 第一个全连接层
                inp=input_feat["fc1"], # 输入特征
            )
        )
        # fc2 (第二个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.fc1, # 前一个操作是第一个全连接层
                layers=[module.fc2], # 第二个全连接层
                inp=input_feat["fc2"], # 输入特征
            )
        )

    elif isinstance(module, LlamaDecoderLayer):
        # 注意力输入
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm, # 前一个操作是输入层归一化
                layers=[
                    module.self_attn.q_proj, # Q投影层
                    module.self_attn.k_proj, # K投影层
                    module.self_attn.v_proj, # V投影层
                ],
                inp=input_feat["self_attn.q_proj"], # 输入特征
                reweight_ratio=reweight_ratio_dict["attn"], # 注意力重加权比例
                module2inspect=module.self_attn, # 检查自注意力模块的输出差异
                kwargs=module_kwargs, # 关键字参数
            )
        )
        # 注意力输出
        # 请参考 https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape: # 如果V投影层和O投影层的权重形状相同
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attn.v_proj, # 前一个操作是V投影层
                    layers=[module.self_attn.o_proj], # 输出投影层
                    inp=input_feat["self_attn.o_proj"], # 输入特征
                    reweight_ratio=reweight_ratio_dict["attn"], # 注意力重加权比例
                )
            )
        # fc1 (第一个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm, # 前一个操作是注意力后层归一化
                layers=[module.mlp.gate_proj, module.mlp.up_proj], # gate投影层和up投影层
                inp=input_feat["mlp.gate_proj"], # 输入特征
                reweight_ratio=reweight_ratio_dict["mlp"], # MLP重加权比例
                module2inspect=module.mlp, # 检查MLP模块的输出差异
            )
        )
        # fc2 (第二个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.up_proj, # 前一个操作是up投影层
                layers=[module.mlp.down_proj], # down投影层
                inp=input_feat["mlp.down_proj"], # 输入特征
                reweight_ratio=reweight_ratio_dict["mlp"], # MLP重加权比例
            )
        )

    elif isinstance(module, BloomBlock):
        # 注意力输入
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm, # 前一个操作是输入层归一化
                layers=[module.self_attention.query_key_value], # query_key_value层
                inp=input_feat["self_attention.query_key_value"], # 输入特征
                module2inspect=module, # 检查当前模块的输出差异
                kwargs=module_kwargs, # 关键字参数
            )
        )
        # 注意力输出
        # 请参考 https://github.com/mit-han-lab/llm-awq/issues/2#issuecomment-1606297469
        """
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # fc1 (第一个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm, # 前一个操作是注意力后层归一化
                layers=[module.mlp.dense_h_to_4h], # dense_h_to_4h层
                inp=input_feat["mlp.dense_h_to_4h"], # 输入特征
                module2inspect=module, # 检查当前模块的输出差异
                kwargs=module_kwargs, # 关键字参数
            )
        )
        # fc2 (第二个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.gelu_impl, # 前一个操作是GELU激活函数
                layers=[module.mlp.dense_4h_to_h], # dense_4h_to_h层
                inp=input_feat["mlp.dense_4h_to_h"], # 输入特征
            )
        )
    elif "mpt" in str(module.__class__).lower():
        # 注意力输入
        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm_1, # 前一个操作是norm_1
                layers=[module.attn.Wqkv], # Wqkv层
                inp=input_feat["attn.Wqkv"], # 输入特征
                module2inspect=module.attn, # 检查注意力模块的输出差异
                kwargs=module_kwargs, # 关键字参数
            )
        )

        # 注意力输出
        scales_list.append(
            _auto_get_scale(
                prev_op=module.attn.Wqkv, # 前一个操作是Wqkv层
                layers=[module.attn.out_proj], # out_proj层
                inp=input_feat["attn.out_proj"], # 输入特征
            )
        )
        # fc1 (第一个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.norm_2, # 前一个操作是norm_2
                layers=[module.ffn.up_proj], # up_proj层
                inp=input_feat["ffn.up_proj"], # 输入特征
                module2inspect=module.ffn, # 检查FFN模块的输出差异
            )
        )
        # fc2 (第二个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ffn.act, # 前一个操作是FFN激活函数
                layers=[module.ffn.down_proj], # down_proj层
                inp=input_feat["ffn.down_proj"], # 输入特征
            )
        )

    elif "falcon" in str(module.__class__).lower():
        # 注意力输出
        # Haotian: TBD: 需要处理MQ的重复尺度
        """
        scales_list.append(_auto_get_scale(
            prev_op=module.self_attention.query_key_value,
            layers=[module.self_attention.dense],
            inp=input_feat['self_attention.dense'],
        ))
        """
        # fc1 (第一个全连接层)，只要它被缩放，一切都会搞砸
        if "falcon-7b" in str(module.__class__).lower():
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.input_layernorm, # 前一个操作是输入层归一化
                    layers=[
                        module.mlp.dense_h_to_4h, # dense_h_to_4h层
                        module.self_attention.query_key_value, # query_key_value层
                    ],
                    inp=input_feat["self_attention.query_key_value"], # 输入特征
                    module2inspect=module, # 检查当前模块的输出差异
                    kwargs=module_kwargs, # 关键字参数
                )
            )
        elif "falcon-40b" in str(module.__class__).lower():
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.ln_attn, # 前一个操作是注意力层归一化
                    layers=[module.self_attention.query_key_value], # query_key_value层
                    inp=input_feat["self_attention.query_key_value"], # 输入特征
                    module2inspect=module, # 检查当前模块的输出差异
                    kwargs=module_kwargs, # 关键字参数
                )
            )
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.ln_mlp, # 前一个操作是MLP层归一化
                    layers=[module.mlp.dense_h_to_4h], # dense_h_to_4h层
                    inp=input_feat["mlp.dense_h_to_4h"], # 输入特征
                    module2inspect=module, # 检查当前模块的输出差异
                    kwargs=module_kwargs, # 关键字参数
                )
            )
        else:
            raise NotImplementedError(
                "未知的Falcon架构，目前仅支持falcon-7b和falcon-40b"
            )
        # fc2 (第二个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act, # 前一个操作是MLP激活函数
                layers=[module.mlp.dense_4h_to_h], # dense_4h_to_h层
                inp=input_feat["mlp.dense_4h_to_h"], # 输入特征
            )
        )
    elif "bigcode" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ln_1, # 前一个操作是ln_1
                layers=[module.attn.c_attn], # c_attn层
                inp=input_feat["attn.c_attn"], # 输入特征
                module2inspect=module.attn, # 检查注意力模块的输出差异
                kwargs=module_kwargs, # 关键字参数
            )
        )
        # fc1 (第一个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ln_2, # 前一个操作是ln_2
                layers=[module.mlp.c_fc], # c_fc层
                inp=input_feat["mlp.c_fc"], # 输入特征
                module2inspect=module.mlp, # 检查MLP模块的输出差异
            )
        )
        # fc2 (第二个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act, # 前一个操作是MLP激活函数
                layers=[module.mlp.c_proj], # c_proj层
                inp=input_feat["mlp.c_proj"], # 输入特征
            )
        )
    elif "neox" in str(module.__class__).lower():
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm, # 前一个操作是输入层归一化
                layers=[module.attention.query_key_value], # query_key_value层
                inp=input_feat["attention.query_key_value"], # 输入特征
                module2inspect=module.attention, # 检查注意力模块的输出差异
                kwargs=module_kwargs, # 关键字参数
            )
        )
        # fc1 (第一个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm, # 前一个操作是注意力后层归一化
                layers=[module.mlp.dense_h_to_4h], # dense_h_to_4h层
                inp=input_feat["mlp.dense_h_to_4h"], # 输入特征
                module2inspect=module.mlp, # 检查MLP模块的输出差异
            )
        )
        # fc2 (第二个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.act, # 前一个操作是MLP激活函数
                layers=[module.mlp.dense_4h_to_h], # dense_4h_to_h层
                inp=input_feat["mlp.dense_4h_to_h"], # 输入特征
            )
        )
    elif module.__class__.__name__ == "Qwen2DecoderLayer":
        # 注意力输入
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm, # 前一个操作是输入层归一化
                layers=[
                    module.self_attn.q_proj, # Q投影层
                    module.self_attn.k_proj, # K投影层
                    module.self_attn.v_proj, # V投影层
                ],
                inp=input_feat["self_attn.q_proj"], # 输入特征
                reweight_ratio=reweight_ratio_dict["attn"], # 注意力重加权比例
                module2inspect=module.self_attn, # 检查自注意力模块的输出差异
                kwargs=module_kwargs, # 关键字参数
            )
        )
        # 注意力输出
        # 请参考 https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape: # 如果V投影层和O投影层的权重形状相同
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attn.v_proj, # 前一个操作是V投影层
                    layers=[module.self_attn.o_proj], # 输出投影层
                    inp=input_feat["self_attn.o_proj"], # 输入特征
                    reweight_ratio=reweight_ratio_dict["attn"], # 注意力重加权比例
                )
            )
        # fc1 (第一个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm, # 前一个操作是注意力后层归一化
                layers=[module.mlp.gate_proj, module.mlp.up_proj], # gate投影层和up投影层
                inp=input_feat["mlp.gate_proj"], # 输入特征
                reweight_ratio=reweight_ratio_dict["mlp"], # MLP重加权比例
                module2inspect=module.mlp, # 检查MLP模块的输出差异
            )
        )
        # fc2 (第二个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.up_proj, # 前一个操作是up投影层
                layers=[module.mlp.down_proj], # down投影层
                inp=input_feat["mlp.down_proj"], # 输入特征
                reweight_ratio=reweight_ratio_dict["mlp"], # MLP重加权比例
            )
        )
    elif module.__class__.__name__ == "InternLM2DecoderLayer":
        # 注意力输入
        scales_list.append(
            _auto_get_scale(
                prev_op=module.attention_norm, # 前一个操作是注意力归一化
                layers=[
                    module.attention.wqkv, # wqkv层
                ],
                inp=input_feat["attention.wqkv"], # 输入特征
                reweight_ratio=reweight_ratio_dict["attn"], # 注意力重加权比例
                module2inspect=module.attention, # 检查注意力模块的输出差异
                kwargs=module_kwargs, # 关键字参数
            )
        )
        # attn out
        # 请参考 https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.attention.wqkv.weight.shape == module.attention.wo.weight.shape: # 如果wqkv层和wo层的权重形状相同
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.attention.wqkv, # 前一个操作是wqkv层
                    layers=[module.attention.wo], # wo层
                    inp=input_feat["attention.wo"], # 输入特征
                    reweight_ratio=reweight_ratio_dict["attn"], # 注意力重加权比例
                )
            )
        # fc1 (第一个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.ffn_norm, # 前一个操作是FFN归一化
                layers=[module.feed_forward.w1, module.feed_forward.w3], # w1层和w3层
                inp=input_feat["feed_forward.w1"], # 输入特征
                reweight_ratio=reweight_ratio_dict["mlp"], # MLP重加权比例
                module2inspect=module.feed_forward, # 检查前馈网络模块的输出差异
            )
        )
        # fc2 (第二个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.feed_forward.w3, # 前一个操作是w3层
                layers=[module.feed_forward.w2], # w2层
                inp=input_feat["feed_forward.w2"], # 输入特征
                reweight_ratio=reweight_ratio_dict["mlp"], # MLP重加权比例
            )
        )
    
    elif module.__class__.__name__ == "Qwen2VLDecoderLayer":
        # attention input
        scales_list.append(
            _auto_get_scale(
                prev_op=module.input_layernorm, # 前一个操作是输入层归一化
                layers=[
                    module.self_attn.q_proj, # Q投影层
                    module.self_attn.k_proj, # K投影层
                    module.self_attn.v_proj, # V投影层
                ],
                inp=input_feat["self_attn.q_proj"], # 输入特征
                reweight_ratio=reweight_ratio_dict["attn"], # 注意力重加权比例
                module2inspect=module.self_attn, # 检查自注意力模块的输出差异
                kwargs=module_kwargs, # 关键字参数
            )
        )
        # attn out
        # 请参考 https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape: # 如果V投影层和O投影层的权重形状相同
            scales_list.append(
                _auto_get_scale(
                    prev_op=module.self_attn.v_proj, # 前一个操作是V投影层
                    layers=[module.self_attn.o_proj], # 输出投影层
                    inp=input_feat["self_attn.o_proj"], # 输入特征
                    reweight_ratio=reweight_ratio_dict["attn"], # 注意力重加权比例
                )
            )
        # fc1 (第一个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.post_attention_layernorm, # 前一个操作是注意力后层归一化
                layers=[module.mlp.gate_proj, module.mlp.up_proj], # gate投影层和up投影层
                inp=input_feat["mlp.gate_proj"], # 输入特征
                reweight_ratio=reweight_ratio_dict["mlp"], # MLP重加权比例
                module2inspect=module.mlp, # 检查MLP模块的输出差异
            )
        )
        # fc2 (第二个全连接层)
        scales_list.append(
            _auto_get_scale(
                prev_op=module.mlp.up_proj, # 前一个操作是up投影层
                layers=[module.mlp.down_proj], # down投影层
                inp=input_feat["mlp.down_proj"], # 输入特征
                reweight_ratio=reweight_ratio_dict["mlp"], # MLP重加权比例
            )
        )
    else:
        raise NotImplementedError(f"prev_op {type(module)} not supported yet!") # 如果模块类型未被支持，则抛出错误

    return scales_list


def apply_scale(module, scales_list, input_feat_dict=None):
    """
    将计算出的尺度因子应用到模型中的相应层。

    Args:
        module (nn.Module): 待应用尺度的模型。
        scales_list (list): 包含尺度信息的列表，每个元素是一个元组 (prev_op_name, layer_names, scales)。
        input_feat_dict (dict, optional): 包含输入特征的字典，如果提供，也会对输入特征进行尺度调整。默认为None。
    """
    for prev_op_name, layer_names, scales in scales_list: # 遍历尺度列表
        prev_op = get_op_by_name(module, prev_op_name) # 获取前一个操作模块
        layers = [get_op_by_name(module, name) for name in layer_names] # 获取需要缩放的层

        prev_op.cuda() # 将前一个操作移到CUDA
        for layer in layers:
            layer.cuda() # 将所有层移到CUDA
        scales.cuda() # 将尺度移到CUDA

        if isinstance(prev_op, nn.Linear): # 如果前一个操作是线性层
            assert len(layers) == 1, "如果prev_op是nn.Linear，则layers必须只有一个元素。"
            scale_fc_fc(prev_op, layers[0], scales) # 对两个全连接层进行尺度调整
        elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm)) or prev_op.__class__.__name__ == "InternLM2RMSNorm" or prev_op.__class__.__name__ == "Qwen2RMSNorm": # 如果前一个操作是LayerNorm或LlamaRMSNorm
            scale_ln_fcs(prev_op, layers, scales) # 对LayerNorm和全连接层进行尺度调整
        elif isinstance(prev_op, (nn.GELU, BloomGelu, GELUActivation)): # 如果前一个操作是GELU激活函数
            new_module = ScaledActivation(prev_op, scales) # 创建ScaledActivation模块
            set_op_by_name(module, prev_op_name, new_module) # 替换原始模块
            scale_gelu_fc(prev_op, layers[0], scales) # 对GELU和全连接层进行尺度调整
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!") # 如果前一个操作类型未被支持，则抛出错误

        # 如果提供了input_feat_dict，则对输入特征应用尺度
        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name] # 获取输入特征
                inp.div_(scales.view(1, -1).to(inp.device)) # 输入特征除以尺度

        prev_op.cpu() # 将前一个操作移回CPU
        for layer in layers:
            layer.cpu() # 将所有层移回CPU
        scales.cpu() # 将尺度移回CPU
