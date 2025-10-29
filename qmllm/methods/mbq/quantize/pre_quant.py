import torch
import torch.nn as nn
import tqdm
import copy
import gc
import functools
from collections import defaultdict
from typing import List

import numpy as np
from torch.nn import CrossEntropyLoss
from transformers.models.bloom.modeling_bloom import BloomForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from qmllm.utils.search import append_str_prefix, get_op_name

from qmllm.methods.mbq.quantize.auto_scale_wa_distort import auto_scale_block_wa_distort
from qmllm.methods.mbq.quantize.auto_scale_wa import auto_scale_block_wa
from qmllm.methods.mbq.quantize.auto_scale_distort import auto_scale_block_distort
from qmllm.methods.mbq.quantize.auto_scale import auto_scale_block, apply_scale
from qmllm.quantization.qlinear import WALinear
from qmllm.quantization.quant_funcs import pseudo_quantize_tensor
from .quantizer import get_module_by_name_suffix


__all__ = ["run_mbq"]


class GradCacheHook:
    """
    用于在模型反向传播过程中缓存特定层梯度的钩子。
    主要用于收集视觉（vis）和文本（cap）部分的梯度信息，
    以便后续分析和重加权。
    """
    def __init__(self, vis_masks, cap_masks):
        """
        初始化GradCacheHook。

        Args:
            vis_masks (torch.Tensor): 视觉部分的掩码。
            cap_masks (torch.Tensor): 文本部分的掩码。
        """
        if vis_masks is None or cap_masks is None:
            raise ValueError("vis_masks 和 cap_masks 不能为空。")
        self.hooks = []  # 存储注册的钩子
        self.vis_masks = vis_masks.cpu()  # 视觉掩码，移到CPU
        self.cap_masks = cap_masks.cpu()  # 文本掩码，移到CPU
        self.steps = {}  # 记录每个模块的步数，用于匹配正确的掩码
        self.grad_dict = {}  # 存储收集到的梯度数据


    def cache_grad_hook(self, module, inp, out, name):
        """
        反向传播钩子函数，用于缓存指定模块的梯度。

        Args:
            module (nn.Module): 注册钩子的模块。
            inp (tuple): 模块输入的梯度。
            out (tuple): 模块输出的梯度。
            name (str): 模块的名称。
        """
        # 初始化步数计数器，用于为梯度找到正确的掩码
        if name not in self.steps:
            self.steps[name] = 0

        # 初始化梯度字典，存储视觉和文本梯度
        if name not in self.grad_dict:
            self.grad_dict[name] = {"vis_grad": [], "cap_grad": []}

        output_grad = out[0].float()  # 获取输出梯度，并转换为float类型
        step = self.steps[name]  # 当前模块的步数

        B, N, C = output_grad.shape  # 获取梯度形状：批次大小、序列长度、特征维度

        for batch_idx in range(B):
            vis_mask = self.vis_masks[step]  # 获取当前步的视觉掩码
            cap_mask = self.cap_masks[step]  # 获取当前步的文本掩码

            vis_grad = output_grad[batch_idx][vis_mask]  # 提取视觉部分的梯度
            cap_grad = output_grad[batch_idx][cap_mask]  # 提取文本部分的梯度

            vis_grad_avg = vis_grad.abs().mean()  # 计算视觉梯度绝对值的平均值
            cap_grad_avg = cap_grad.abs().mean()  # 计算文本梯度绝对值的平均值

            # 将平均梯度添加到字典中
            self.grad_dict[name]["vis_grad"].append(vis_grad_avg.detach().cpu())
            self.grad_dict[name]["cap_grad"].append(cap_grad_avg.detach().cpu())

            step = step + 1  # 步数递增

        self.steps[name] = step  # 更新模块的步数


    def register_hooks(self, layers):
        """
        为指定层中的线性模块注册反向传播钩子。

        Args:
            layers (nn.ModuleList): 包含模型层的列表。
        """
        for n, m in layers.named_modules():
            # 检查是否是nn.Linear层，并且名称包含特定的关键字（通常是FFN或Attention的输出投影层）
            if isinstance(m, nn.Linear) and any([_ in n for _ in ["wo", "w2", "down_proj", "o_proj", "v_proj", "gate_proj", "up_proj", "w1", "w3"]]):
                # print(f"Registering hook for layer.{n}") # 调试信息
                self.hooks.append(
                    m.register_full_backward_hook(
                        functools.partial(self.cache_grad_hook, name=f"layers.{n}") # 注册钩子，并绑定模块名称
                    )
                )


    def remove_hooks(self):
        """
        移除所有已注册的钩子。
        """
        for h in self.hooks:
            h.remove()  # 移除单个钩子
        self.hooks.clear()  # 清空钩子列表


    def get_grad_dict(self):
        """
        获取存储所有模块梯度的字典。

        Returns:
            dict: 包含每个模块视觉和文本梯度的字典。
        """
        return self.grad_dict
    

    def get_avg_grad_dict(self):
        """
        计算并获取每个模块的平均梯度字典。

        Returns:
            dict: 包含每个模块视觉和文本平均梯度的字典。
        """
        avg_grad_dict = {}

        for name, grad_values in self.grad_dict.items():
            # 计算视觉梯度的平均值
            mean_vis = torch.mean(torch.stack(grad_values["vis_grad"]))
            # 计算文本梯度的平均值
            mean_cap = torch.mean(torch.stack(grad_values["cap_grad"]))

            avg_grad_dict[name] = {
                "vis_avg_grad": mean_vis.item(),  # 视觉平均梯度
                "cap_avg_grad": mean_cap.item()   # 文本平均梯度
            }

        return avg_grad_dict
    

def get_named_linears(module):
    """
    获取模块中所有命名线性层的字典。

    Args:
        module (nn.Module): 要检查的PyTorch模块。

    Returns:
        dict: 键为线性层名称，值为线性层模块的字典。
    """
    # 遍历模块的所有命名子模块，筛选出nn.Linear类型的模块
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_blocks(model):
    """
    根据模型类型获取其Transformer块（层）的列表。

    Args:
        model (nn.Module): 待处理的模型。

    Returns:
        nn.ModuleList: 包含模型所有Transformer块的列表。

    Raises:
        NotImplementedError: 如果模型类型未被支持。
    """
    # 根据不同的模型类名，获取其对应的层列表
    if model.__class__.__name__ == "LlamaForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        # 对于LlavaLlamaForCausalLM模型，获取其Llama层的列表
        # layers = [model.model.layers, model.model.vision_tower.vision_tower.vision_model.encoder.layers] # 原始注释，可能包含视觉塔层
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaQwenForCausalLM":
        # 对于LlavaQwenForCausalLM模型，获取其Qwen层的列表
        layers = model.model.layers
    elif model.__class__.__name__ == "InternLM2ForCausalLM":
        # 对于InternLM2ForCausalLM模型，获取其InternLM2层的列表
        layers = model.model.layers
    elif model.__class__.__name__ == "InternVLChatModel":
        # 对于InternVLChatModel模型，获取其语言模型的层列表
        layers = model.language_model.model.layers
    elif model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
        # 对于Qwen2VLForConditionalGeneration模型，获取其Qwen2层的列表
        layers = model.model.layers
    elif model.__class__.__name__ == "LlavaLlamaModel":
        # 对于LlavaLlamaModel模型，获取其LLM部分的层列表
        layers = model.llm.model.layers
    elif isinstance(model, OPTForCausalLM):
        # 对于OPTForCausalLM模型，获取其解码器层的列表
        layers = model.model.decoder.layers
    elif isinstance(model, BloomForCausalLM):
        # 对于BloomForCausalLM模型，获取其Transformer块的列表
        layers = model.transformer.h
    elif "mpt" in str(model.__class__).lower():
        # 对于MPT系列模型，获取其Transformer块的列表
        layers = model.transformer.blocks
    elif "falcon" in str(model.__class__).lower():
        # 对于Falcon系列模型，获取其Transformer块的列表
        layers = model.transformer.h
    elif "bigcode" in str(model.__class__).lower():
        # 对于BigCode系列模型，获取其Transformer块的列表
        layers = model.transformer.h
    elif "neox" in str(model.__class__).lower():
        # 对于NeoX系列模型，获取其GPT-NeoX层的列表
        layers = model.gpt_neox.layers
    else:
        # 如果模型类型未被支持，则抛出NotImplementedError
        raise NotImplementedError(type(model))
    return layers


def move_embed(model, device):
    """
    将模型的嵌入层（embedding layers）移动到指定的设备。

    Args:
        model (nn.Module): 待处理的模型。
        device (str or torch.device): 目标设备（例如 "cuda" 或 "cpu"）。

    Raises:
        NotImplementedError: 如果模型类型未被支持。
    """
    # 根据不同的模型类型，将其嵌入层移动到指定设备
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device) # Llama的词嵌入层
        model.model.rotary_emb = model.model.rotary_emb.to(device) # Llama的旋转嵌入层
    elif isinstance(model, OPTForCausalLM):
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(device) # OPT的词嵌入层
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to( # OPT的位置嵌入层
            device
        )
    elif isinstance(model, BloomForCausalLM):
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device) # Bloom的词嵌入层
        model.transformer.word_embeddings_layernorm = ( # Bloom的词嵌入层归一化
            model.transformer.word_embeddings_layernorm.to(device)
        )
    elif "mpt" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device) # MPT的词嵌入层
        model.transformer.emb_drop = model.transformer.emb_drop.to(device) # MPT的嵌入层dropout
    elif "falcon" in str(model.__class__).lower():
        model.transformer.word_embeddings = model.transformer.word_embeddings.to(device) # Falcon的词嵌入层
    elif "bigcode" in str(model.__class__).lower():
        model.transformer.wte = model.transformer.wte.to(device) # BigCode的词嵌入层
        model.transformer.wpe = model.transformer.wpe.to(device) # BigCode的位置嵌入层
        model.transformer.drop = model.transformer.drop.to(device) # BigCode的dropout层
    elif "neox" in str(model.__class__).lower():
        model.gpt_neox.embed_in = model.gpt_neox.embed_in.to(device) # NeoX的输入嵌入层
        model.gpt_neox.emb_dropout = model.gpt_neox.emb_dropout.to(device) # NeoX的嵌入层dropout
        model.embed_out = model.embed_out.to(device) # NeoX的输出嵌入层
    elif model.__class__.__name__ == "LlavaLlamaForCausalLM":
        model.model.embed_tokens = model.model.embed_tokens.to(device) # LlavaLlama的词嵌入层
        model.model.vision_tower.vision_tower.vision_model.embeddings.to(device) # LlavaLlama的视觉塔嵌入层
    elif model.__class__.__name__ == "LlavaQwenForCausalLM":
        model.model.embed_tokens = model.model.embed_tokens.to(device) # LlavaQwen的词嵌入层
        # model.model.rotary_emb = model.model.rotary_emb.to(device) # 原始注释，可能包含旋转嵌入层
    elif model.__class__.__name__ == "InternLM2ForCausalLM":
        model.model.tok_embeddings = model.model.tok_embeddings.to(device) # InternLM2的token嵌入层
    elif model.__class__.__name__ == "InternVLChatModel":
        model.language_model.model.tok_embeddings = model.language_model.model.tok_embeddings.to(device)  # InternVLChat的语言模型token嵌入层
    elif model.__class__.__name__ == "Qwen2VLForConditionalGeneration":
        model.model.embed_tokens = model.model.embed_tokens.to(device) # Qwen2VL的词嵌入层
    elif model.__class__.__name__ == "LlavaLlamaModel":
        model.llm.model.embed_tokens = model.llm.model.embed_tokens.to(device) # LlavaLlamaModel的LLM词嵌入层
    else:
        raise NotImplementedError(type(model)) # 如果模型类型未被支持，则抛出错误


def process_input(prompt_inputs, prompt_kwargs):
    """
    处理输入，合并prompt_inputs和prompt_kwargs，并提取vision_mask和caption_mask。

    Args:
        prompt_inputs (dict): 包含模型输入数据的字典。
        prompt_kwargs (dict): 包含模型额外关键字参数的字典。

    Returns:
        tuple: 包含处理后的输入字典、视觉掩码和文本掩码。
    """
    inputs = {**prompt_inputs, **prompt_kwargs} # 合并输入字典
    inputs["use_cache"] = False # 禁用缓存
    vision_mask = inputs.pop("vision_mask", None) # 提取并移除视觉掩码
    caption_mask = inputs.pop("caption_mask", None) # 提取并移除文本掩码
    
    return inputs, vision_mask, caption_mask # 返回处理后的输入和掩码


@torch.no_grad()
def run_mbq(
    model,
    prompt_inputs,
    prompt_kwargs,
    w_bit,
    a_bit,
    q_config,
    auto_scale=True,
    loss_mode="mae",
    wa_quant=False,
    reweight=False,
    distort=False
):
    """
    运行MBQ (Multi-Bit Quantization) 预量化过程，计算并返回量化参数。

    此函数在无梯度模式下运行，通过逐层处理模型，收集输入特征，
    并根据不同的量化策略（如自动缩放、重加权、失真）计算量化尺度。

    Args:
        model (nn.Module): 待量化的原始模型。
        prompt_inputs (dict): 模型校准或处理所需的输入。
        prompt_kwargs (dict): 模型校准或处理所需的额外关键字参数。
        w_bit (int): 权重的位宽。
        a_bit (int): 激活的位宽。
        q_config (dict): 量化配置字典，包含零点和分组大小等。
        auto_scale (bool): 是否自动计算量化尺度。
        loss_mode (str): 量化过程中的损失计算模式，默认为"mae"。
        wa_quant (bool): 是否同时量化权重和激活。
        reweight (bool): 是否进行重加权。
        distort (bool): 是否进行失真处理。

    Returns:
        dict: 包含量化尺度等MBQ结果的字典。
    """
    # 如果模型是"bigcode"系列，将attention_mask的偏置项移动到CUDA
    if "bigcode" in str(model.model.__class__).lower():
        # 否则attention_mask将始终在CPU上。
        model.transformer.bias = model.transformer.bias.to("cuda")

    # 获取模型的所有Transformer块（层）
    layers = get_blocks(model.model)

    inps = []  # 存储第一层的输入
    layer_kwargs = {}  # 存储第一层的关键字参数

    # 将第一层移动到CUDA设备
    layers[0] = layers[0].cuda()
    # 将模型的嵌入层移动到CUDA设备
    move_embed(model.model, "cuda")

    # 获取第0层的输入和关键字参数
    # PyTorch 2.0才支持with_kwargs
    # 暂时使用Catcher hack来捕获输入和kwargs
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)  # 捕获输入
            layer_kwargs.update(kwargs)  # 捕获关键字参数
            raise ValueError  # 提前退出，中断后续推理

    # 打补丁到第0层以捕获输入和kwargs
    layers[0] = Catcher(layers[0])

    # 处理输入，获取inputs、vision_mask和caption_mask
    inputs, vision_mask, caption_mask = process_input(prompt_inputs, prompt_kwargs)

    model.to_cuda() # 将整个模型移动到CUDA
    try:
        model(**inputs) # 运行模型前向传播，触发Catcher的ValueError
    except ValueError: # 捕获ValueError，继续执行
        pass

    model.to_cpu() # 将模型移回CPU
    layers[0] = layers[0].module  # 恢复第0层原始模块
    inps = inps[0]  # 获取捕获到的第一层输入
    layer_kwargs["use_cache"] = False  # 禁用缓存

    layers[0] = layers[0].cpu() # 将第0层移回CPU
    move_embed(model.model, "cpu") # 将嵌入层移回CPU

    gc.collect() # 垃圾回收
    torch.cuda.empty_cache() # 清空CUDA缓存

    mbq_results = {
        "scale": [], # 存储量化尺度
    }

    # 如果需要重加权（reweight）：即使用基于梯度的敏感度方法
    if reweight:
        model.to_cuda() # 将模型移到CUDA
        print("Save gradient...")
        # 保存梯度
        grad_cache = GradCacheHook(vis_masks=vision_mask, cap_masks=caption_mask) # 初始化梯度缓存钩子
        grad_cache.register_hooks(layers=layers) # 为所有层注册钩子
        
        with torch.enable_grad(): # 启用梯度计算
            mini_batch = 1 # 小批量大小
            total_samples = next(iter(prompt_inputs.values())).shape[0] # 总样本数
            accum_steps = int(total_samples/mini_batch) # 梯度累积步数
            
            # 遍历小批量数据进行梯度计算
            for i in tqdm.tqdm(range(0, total_samples, mini_batch), desc="Running gradient calculation..."):
                mini_inputs = {}
                for k in inputs:
                    if isinstance(inputs[k], torch.Tensor):
                        mini_inputs[k] = inputs[k][i:i+mini_batch] # 准备小批量输入
                
                outputs = model(**mini_inputs) # 模型前向传播

                loss = outputs[0] # 获取损失

                loss = loss / accum_steps # 损失归一化
                loss.backward() # 反向传播

        model.to_cpu() # 将模型移回CPU
        grad_avg_dict = grad_cache.get_avg_grad_dict() # 获取平均梯度字典
        grad_cache.remove_hooks() # 移除钩子
        del grad_cache # 删除梯度缓存
        
        attn_list = [] # 存储注意力层的重加权比例
        mlp_list = [] # 存储MLP层的重加权比例

        # 遍历平均梯度字典，计算注意力层和MLP层的重加权比例
        for key_name in grad_avg_dict:
            if "down_" in key_name or "w2" in key_name: # MLP层
                mlp_list.append(grad_avg_dict[key_name]["vis_avg_grad"] / grad_avg_dict[key_name]["cap_avg_grad"])
            if "o_proj" in key_name or "wo" in key_name: # 注意力层
                attn_list.append(grad_avg_dict[key_name]["vis_avg_grad"] / grad_avg_dict[key_name]["cap_avg_grad"])

        attn_median = np.median(attn_list) # 注意力层重加权比例的中位数
        mlp_median = np.median(mlp_list) # MLP层重加权比例的中位数

    # 如果需要失真（distort）
    if distort:
        # assert wa_quant, "We only support distort input in weight-activation quantization!!!" # 断言，仅在权重-激活量化时支持失真输入
        print("Use distort input...")
        inps_distort = copy.deepcopy(inps) # 复制输入用于失真处理

    gc.collect() # 垃圾回收
    torch.cuda.empty_cache() # 清空CUDA缓存

    # 逐层处理
    for i in tqdm.tqdm(range(len(layers)), desc="Running MBQ..."):
        layer = layers[i] # 获取当前层
        layer = layer.cuda() # 将当前层移到CUDA
        named_linears = get_named_linears(layer) # 获取当前层中的所有线性层

        # 首先，获取所有线性层的输入特征
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu() # 将输入特征移到CPU并分离
            feat_dict[name].append(x) # 缓存输入特征

        input_feat = defaultdict(list) # 存储输入特征的字典
        handles = [] # 存储钩子句柄
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat) # 注册前向钩子
                )
            )
        inps = inps.to(next(layer.parameters()).device)  # 将输入移到当前层的设备（以防多GPU）
        # 获取输出作为下一层的输入

        for k in layer_kwargs:
            if isinstance(layer_kwargs[k], torch.Tensor):
                layer_kwargs[k] = layer_kwargs[k].to(next(layer.parameters()).device) # 将关键字参数移到当前层的设备

        inps = layer(inps, **layer_kwargs)[0] # 模型前向传播，获取输出作为下一层的输入
        for h in handles:
            h.remove() # 移除钩子
        # 现在解决缩放问题
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()} # 合并所有输入特征

        # 清空GPU内存
        torch.cuda.empty_cache()

        if reweight:
            scale_reweight_ratio_dict = {} # 存储重加权比例的字典
            for key, value in grad_avg_dict.items():
                item_list = key.split(".")
                if str(i) in item_list: # 如果当前层包含在梯度字典的键中
                    if "wo" in item_list or "o_proj" in item_list: # 注意力层
                        scale_reweight_ratio_dict["attn"] = max((value["vis_avg_grad"] / value["cap_avg_grad"]), attn_median) # 计算注意力层重加权比例
                    elif "w2" in item_list or "down_proj" in item_list: # MLP层
                        scale_reweight_ratio_dict["mlp"] = max((value["vis_avg_grad"] / value["cap_avg_grad"]), mlp_median) # 计算MLP层重加权比例
        else:
            scale_reweight_ratio_dict = {
                "attn": None,
                "mlp": None
            }

        if (
            auto_scale
        ):  # 如果应用自动缩放，我们也应该用尺度修改input_feat
            if not reweight:
                ans_mask = None
                vis_mask = None
            else:
                ans_mask = caption_mask
                vis_mask = vision_mask
            
            if wa_quant: # 如果进行权重-激活联合量化
                if distort: # 如果进行失真处理
                    scales_list = auto_scale_block_wa_distort( # 自动缩放权重-激活并失真
                        layer,
                        layer_kwargs,
                        w_bit=w_bit,
                        a_bit=a_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        vis_mask=vis_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        q_input=inps_distort,
                        loss_mode=loss_mode
                    )
                else: # 不进行失真处理
                    scales_list = auto_scale_block_wa( # 自动缩放权重-激活
                        layer,
                        layer_kwargs,
                        w_bit=w_bit,
                        a_bit=a_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        vis_mask=vis_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        loss_mode=loss_mode
                    )
            else: # 仅进行权重量化
                if distort: # 如果进行失真处理
                    scales_list = auto_scale_block_distort( # 自动缩放权重并失真
                        layer,
                        layer_kwargs,
                        w_bit=w_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        vis_mask=vis_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        q_input=inps_distort,
                        loss_mode=loss_mode
                    )
                else: # 不进行失真处理
                    scales_list = auto_scale_block( # 自动缩放权重
                        layer,
                        layer_kwargs,
                        w_bit=w_bit,
                        q_config=q_config,
                        input_feat=input_feat,
                        ans_mask=ans_mask,
                        vis_mask=vis_mask,
                        reweight_ratio_dict=scale_reweight_ratio_dict,
                        loss_mode=loss_mode
                    )

            # apply_scale(layer, scales_list, input_feat_dict=input_feat) # 应用尺度到当前层
            apply_scale(layers[i], scales_list, input_feat_dict=input_feat) # 应用尺度到当前层

            if distort: # 如果进行失真处理
                # 获取失真输出作为下一层的输入
                if wa_quant: # 如果进行权重-激活联合量化
                    layer_q = copy.deepcopy(layer) # 复制当前层
                    layer_q = layer_q.cuda() # 将复制层移到CUDA
                    named_linears_q = get_named_linears(layer_q) # 获取复制层中的线性层
                    for n, m in named_linears_q.items():
                        # 创建WALinear层进行权重-激活量化
                        new_linear = WALinear.from_float(m, weight_quant="per_channel", act_quant="per_token", w_bit=w_bit, a_bit=a_bit)
                        father_module = get_module_by_name_suffix(layer_q, '.'.join(n.split(".")[:-1]))
                        setattr(father_module, n.split('.')[-1], new_linear) # 替换原始线性层
                        del new_linear, m
                        torch.cuda.empty_cache()
                    
                    inps_distort = inps_distort.to(next(layer_q.parameters()).device)  # 将失真输入移到当前层的设备
                    inps_distort = layer_q(inps_distort, **layer_kwargs)[0] # 模型前向传播，获取失真输出
                    del layer_q # 删除复制层
                else: # 仅进行权重量化
                    layer_q = copy.deepcopy(layer) # 复制当前层
                    layer_q = layer_q.cuda() # 将复制层移到CUDA
                    named_linears_q = get_named_linears(layer_q) # 获取复制层中的线性层
                    for n, m in named_linears_q.items():
                        m.weight.data = pseudo_quantize_tensor(m.weight.data, n_bits=w_bit, **q_config) # 伪量化权重
                        torch.cuda.empty_cache()
                    
                    inps_distort = inps_distort.to(next(layer_q.parameters()).device)  # 将失真输入移到当前层的设备
                    inps_distort = layer_q(inps_distort, **layer_kwargs)[0] # 模型前向传播，获取失真输出
                    del layer_q # 删除复制层

            # 添加前缀使名称全局化
            mbq_results["scale"] += append_str_prefix(
                scales_list, get_op_name(model.model, layer) + "."
            )

        # 清空GPU内存
        torch.cuda.empty_cache()

        layer = layer.cpu() # 将当前层移回CPU
        # Haotian: 检查激活替换
        del input_feat # 删除输入特征
        gc.collect() # 垃圾回收
        torch.cuda.empty_cache() # 清空CUDA缓存

    return mbq_results # 返回MBQ结果


def apply_mbq(model, mbq_results):
    """
    将MBQ量化结果（尺度因子）应用到模型。

    Args:
        model (nn.Module): 待应用量化参数的模型。
        mbq_results (dict): 包含量化尺度等MBQ结果的字典。
    """
    apply_scale(model, mbq_results["scale"]) # 应用尺度到模型
