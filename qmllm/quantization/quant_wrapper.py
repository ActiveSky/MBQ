import os

# 导入不同的量化方法入口函数
from qmllm.methods.awq.entry import awq_entry
from qmllm.methods.smoothquant.entry import smoothquant_entry
from qmllm.methods.mbq.entry import mbq_entry
from qmllm.methods.rtn.entry import rtn_entry

def qwrapper(model, prompt_inputs, prompt_kwargs, args):
    """
    量化包装器，根据指定的量化方法对模型进行量化。

    根据 `args.method` 参数选择并应用不同的量化策略，
    例如 AWQ, SmoothQuant, MBQ 或 RTN。

    Args:
        model: 待量化的原始模型。
        prompt_inputs: 模型所需的输入数据，用于量化校准或处理。
        prompt_kwargs: 模型所需的额外关键字参数。
        args: 包含量化配置参数的对象，例如量化方法、位宽、组大小等。

    Returns:
        经过量化处理后的模型。

    Raises:
        NotImplementedError: 如果指定的量化方法未实现。
    """
    # 根据命令行参数args.method选择不同的量化方法
    if args.method == "awq":
        # 应用 AWQ (Activation-aware Weight Quantization) 量化方法
        # run_awq_process: 是否运行AWQ处理过程
        # scale_path: 存储量化尺度因子的路径
        # q_group_size: 量化组的大小
        # w_bit: 权重的位宽
        model = awq_entry(model, prompt_inputs, prompt_kwargs, run_awq_process=args.run_process, scale_path=args.scale_path, q_group_size=args.w_group, w_bit=args.w_bit)
    elif args.method == "smoothquant":
        # 应用 SmoothQuant 量化方法
        # run_sq_process: 是否运行SmoothQuant处理过程
        # pseudo_quant: 是否使用伪量化（用于模拟量化效果而非实际量化）
        # scale_path: 存储量化尺度因子的路径
        # w_bit: 权重的位宽
        # a_bit: 激活的位宽
        # alpha: SmoothQuant中的平滑因子
        model = smoothquant_entry(model, prompt_inputs, prompt_kwargs, run_sq_process=args.run_process, pseudo_quant=args.pseudo_quant, scale_path=args.scale_path, w_bit=args.w_bit, a_bit=args.a_bit, alpha=args.alpha)
    elif args.method == "mbq":
        # 应用 MBQ (Multi-Bit Quantization) 量化方法
        # 检查是否同时量化权重和激活（当位宽小于16时）
        wa_quant = args.w_bit < 16 and args.a_bit < 16
        model = mbq_entry(model, prompt_inputs, prompt_kwargs, 
                                run_mbq_process=args.run_process, # 是否运行MBQ处理过程
                                pseudo_quant=args.pseudo_quant, # 是否使用伪量化
                                scale_path=args.scale_path, # 存储量化尺度因子的路径
                                q_group_size=args.w_group, # 量化组的大小
                                w_bit=args.w_bit, # 权重的位宽
                                a_bit=args.a_bit, # 激活的位宽
                                wa_quant=wa_quant, # 是否同时量化权重和激活
                                reweight=args.reweight, # 是否进行重新加权
                                distort=args.distort, # 是否进行失真处理
                                loss_mode=args.loss_mode) # 损失模式
    elif args.method == "rtn":
        # 应用 RTN (Round-to-Nearest) 量化方法
        # 检查是否同时量化权重和激活（当位宽小于16时）
        wa_quant = args.w_bit < 16 and args.a_bit < 16
        model = rtn_entry(model, pseudo_quant=args.pseudo_quant, wa_quant=wa_quant, q_group_size=args.w_group, w_bit=args.w_bit, a_bit=args.a_bit)
    else:
        # 如果方法未识别，则抛出未实现错误
        raise NotImplementedError

    return model # 返回量化后的模型
