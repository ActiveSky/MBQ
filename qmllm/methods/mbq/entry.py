import os
import torch

# 从MBQ量化模块导入所需的函数
from qmllm.methods.mbq.quantize.pre_quant import run_mbq, apply_mbq
from qmllm.methods.mbq.quantize.quantizer import pseudo_quantize_model_weight, pseudo_quantize_model_weight_act


def mbq_entry(model, prompt_inputs, prompt_kwargs, run_mbq_process: bool, pseudo_quant: bool, scale_path: str=None, zero_point: bool=True, q_group_size: int=128, w_bit: int=4, a_bit: int=16, wa_quant: bool=False, reweight: bool=False, distort: bool=False, loss_mode: str="mae"):
    """
    MBQ (Multi-Bit Quantization) 量化方法的入口函数。

    根据提供的参数，执行MBQ量化流程。这可能包括运行量化过程、保存/加载量化参数，
    以及应用伪量化到模型的权重或权重与激活。

    Args:
        model: 待量化的原始模型。
        prompt_inputs: 模型校准或处理所需的输入。
        prompt_kwargs: 模型校准或处理所需的额外关键字参数。
        run_mbq_process: 布尔值，指示是否运行MBQ的预量化处理过程。
        pseudo_quant: 布尔值，指示是否应用伪量化（用于模拟量化效果）。
        scale_path: 字符串，用于保存或加载量化尺度参数的路径。
        zero_point: 布尔值，量化的零点配置，默认为True。
        q_group_size: 整型，量化组的大小，默认为128，表示进行分组量化。
        w_bit: 整型，权重的位宽，默认为4位。
        a_bit: 整型，激活的位宽，默认为16位。
        wa_quant: 布尔值，指示是否同时量化权重和激活，默认为False。
        reweight: 布尔值，指示是否进行重加权，默认为False。
        distort: 布尔值，指示是否进行失真处理，默认为False。
        loss_mode: 字符串，量化过程中的损失计算模式，默认为"mae"。

    Returns:
        torch.nn.Module: 经过MBQ量化处理后的模型。

    Raises:
        AssertionError: 如果 `scale_path` 为None。
    """
    # 构建量化配置字典
    q_config = {
        "zero_point": zero_point,  # 零点量化设置，默认为True
        "q_group_size": q_group_size,  # 量化组大小，控制是否进行分组量化
    }

    # 确保scale_path已被指定
    assert scale_path is not None, "scale_path 不能为空，用于保存或加载量化参数。"

    # 检查scale_path是否存在，以判断是否需要重新运行MBQ过程
    scale_exist = os.path.exists(scale_path)
    
    # reparameterization (重参数化) 阶段
    # 如果需要运行MBQ处理过程且量化参数文件不存在，则执行MBQ的预量化过程
    if run_mbq_process and not scale_exist:
        model.to_cpu() # 将模型移动到CPU进行处理，以节省GPU内存或避免CUDA相关问题
        # 运行MBQ预量化过程，计算并获取量化结果（如尺度因子、零点等）
        mbq_results = run_mbq(
            model,
            prompt_inputs,
            prompt_kwargs,
            w_bit=w_bit, # 权重位宽
            a_bit=a_bit, # 激活位宽
            q_config=q_config, # 量化配置
            auto_scale=True, # 自动计算尺度
            loss_mode=loss_mode, # 损失模式
            wa_quant=wa_quant, # 是否进行权重-激活联合量化
            reweight=reweight, # 是否进行重加权
            distort=distort, # 是否进行失真处理
        )
        
        # 确保保存路径的目录存在
        dirpath = os.path.dirname(scale_path)
        os.makedirs(dirpath, exist_ok=True)
        
        # 保存MBQ量化结果到指定路径
        torch.save(mbq_results, scale_path)
        print("MBQ results saved at", scale_path)

    # 伪量化 (pseudo_quant) 阶段
    # 如果设置了pseudo_quant为True，即使不运行完整MBQ过程也应用伪量化
    if pseudo_quant:
        # 从磁盘加载MBQ量化结果
        mbq_results = torch.load(scale_path, map_location="cpu")
        # 应用MBQ的量化参数到模型的相应层
        apply_mbq(model.model, mbq_results)

        # 根据wa_quant选择性地进行权重伪量化或权重-激活伪量化
        if not wa_quant:
            # 仅进行权重伪量化
            pseudo_quantize_model_weight(model.model, w_bit=w_bit, q_config=q_config)
        else:
            # 进行权重和激活的伪量化
            pseudo_quantize_model_weight_act(model.model, w_bit=w_bit, a_bit=a_bit)

    model.to_cuda() # 将处理后的模型移回CUDA（如果可用）
    return model # 返回量化后的模型
