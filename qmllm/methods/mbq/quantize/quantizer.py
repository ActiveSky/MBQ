import torch
import torch.nn as nn
from tqdm import tqdm
import gc
from qmllm.methods.mbq.quantize.qmodule import ScaledActivation
from qmllm.utils.search import set_op_by_name

from transformers.models.bloom.modeling_bloom import BloomBlock
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from qmllm.quantization.qlinear import WALinear

EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]


def scale_activations(module):
    """
    为特定模型块中的激活函数添加 ScaledActivation 包装器。
    这允许在量化过程中对激活进行动态缩放。

    Args:
        module (torch.nn.Module): 要修改的模型块。
    """
    param = next(module.parameters()) # 获取模块的第一个参数以确定数据类型和设备
    dtype = param.dtype
    device = param.device
    if isinstance(module, BloomBlock):
        # 处理 BloomBlock 模块
        if isinstance(module.mlp.gelu_impl, ScaledActivation):
            return # 如果已经包装过，则直接返回
        c = module.mlp.dense_h_to_4h.out_features # 获取输出特征维度
        act = ScaledActivation(
            module.mlp.gelu_impl, torch.ones(c, dtype=dtype, device=device) # 创建 ScaledActivation 实例
        )
        set_op_by_name(module, "mlp.gelu_impl", act) # 替换原始激活函数
    elif "mptblock" in str(module.__class__.__name__).lower():
        # 处理 MPTBlock 模块
        if isinstance(module.ffn.act, ScaledActivation):
            return
        c = module.ffn.up_proj.out_features
        act = ScaledActivation(
            module.ffn.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "ffn.act", act)
    elif "falcon" in str(module.__class__).lower():
        # 处理 Falcon 模型块
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "bigcode" in str(module.__class__).lower():
        # 处理 BigCode 模型块
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.c_proj.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)
    elif "neox" in str(module.__class__).lower():
        # 处理 NeoX 模型块
        if isinstance(module.mlp.act, ScaledActivation):
            return
        c = module.mlp.dense_h_to_4h.out_features
        act = ScaledActivation(
            module.mlp.act, torch.ones(c, dtype=dtype, device=device)
        )
        set_op_by_name(module, "mlp.act", act)

@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
):
    """
    对模型中的所有线性层进行伪量化（仅权重）。
    该函数遍历模型的每个块，识别其中的线性层，并对其权重数据进行伪量化。

    Args:
        model (torch.nn.Module): 要进行伪量化的模型。
        w_bit (int): 权重的量化位宽。
        q_config (dict): 量化配置参数，例如量化器类型、对称性等。
    """
    from .pre_quant import get_blocks, get_named_linears # 导入辅助函数

    layers = get_blocks(model) # 获取模型的所有块（层）
    for i in tqdm(range(len(layers)), desc="伪权重量化中..."): # 遍历每个块，并显示进度条
        named_linears = get_named_linears(layers[i]) # 获取当前块中所有命名线性层
        for n, m in named_linears.items():
            # 暂时注释掉 CUDA 相关的操作，因为量化可以在 CPU 上进行
            # m.cuda()
            # 对线性层的权重数据进行 **伪量化**(量化->反量化)
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bits=w_bit, **q_config
            )
            # m.cpu() # 将模型移回 CPU (如果之前移到 CUDA)


def get_module_by_name_suffix(model, module_name: str):
    """
    通过名称后缀从模型中获取模块。

    Args:
        model (torch.nn.Module): 要搜索的 PyTorch 模型。
        module_name (str): 要查找的模块的名称后缀。

    Returns:
        torch.nn.Module: 匹配的模块，如果未找到则返回 None。
    """
    for name, module in model.named_modules():
        # 遍历模型中所有命名模块
        if name.endswith(module_name):
            # 如果模块名称以指定的后缀结尾，则返回该模块
            return module
    return None # 如果未找到匹配的模块，则返回 None


@torch.no_grad()
def pseudo_quantize_model_weight_act(
    model,
    w_bit,
    a_bit,
):
    """
    对模型中的所有线性层进行伪量化（权重和激活）。
    该函数遍历模型的每个块，识别其中的线性层，并将其替换为量化后的 WALinear 层。

    Args:
        model (torch.nn.Module): 要进行伪量化的模型。
        w_bit (int): 权重的量化位宽。
        a_bit (int): 激活的量化位宽。
    """
    from .pre_quant import get_blocks, get_named_linears # 导入辅助函数

    layers = get_blocks(model) # 获取模型的所有块（层）
    for i in tqdm(range(len(layers)), desc="伪权重激活量化中..."): # 遍历每个块，并显示进度条
        named_linears = get_named_linears(layers[i]) # 获取当前块中所有命名线性层
        for n, m in named_linears.items():
            # 从浮点线性层创建新的量化 WALinear 层
            new_linear = WALinear.from_float(m, weight_quant="per_channel", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
            # 获取父模块，以便替换原始线性层
            father_module = get_module_by_name_suffix(layers[i], '.'.join(n.split(".")[:-1]))
            # 替换原始线性层为新的量化线性层
            setattr(father_module, n.split('.')[-1], new_linear)
            del new_linear, m # 删除不再需要的对象以释放内存
            torch.cuda.empty_cache() # 清空 CUDA 缓存
