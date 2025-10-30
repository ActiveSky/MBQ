"""
Modality-Balanced Quantization (MBQ) for Large Vision-Language Models (VLMs)
============================================================================

该模块实现了模型量化的主要入口点，支持多种量化方法（如 MBQ、AWQ、SmoothQuant、RTN），
用于对大型视觉-语言模型进行量化处理。

主要功能：
1. 解析命令行参数和配置文件
2. 加载指定的视觉-语言模型
3. 准备校准数据集
4. 执行量化过程
5. 保存量化结果

支持的模型包括：
- InternVL2
- LLaVA-OneVision
- LLaVA-v1.5
- Qwen2-VL
- VILA

支持的量化方法：
- MBQ (Modality-Balanced Quantization)
- AWQ (Activation-aware Weight Quantization)
- SmoothQuant
- RTN (Round-to-Nearest)
"""

import argparse
import os
import warnings
import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

from typing import Union

from lmms_eval.models import get_model

from qmllm.quantization.quant_wrapper import qwrapper
from qmllm.models import get_process_model
from qmllm.calibration.pileval import get_calib_dataset
from qmllm.calibration.coco_vl import get_multimodal_calib_dataset


def parse_quant_args() -> argparse.Namespace:
    """
    解析量化过程所需的命令行参数

    返回:
        argparse.Namespace: 包含所有解析后的命令行参数的对象

    参数说明:
        基础模型参数:
        --config: 指定YAML配置文件路径，如果指定则忽略CLI参数
        --model: 模型名称 (默认: "hf")
        --model_args: 模型参数字符串 (例如: pretrained=EleutherAI/pythia-160m,dtype=float32)
        --batch_size: 批处理大小 (默认: 1)
        --device: 使用的设备 (例如: cuda, cuda:0, cpu)

        校准数据参数:
        --calib_data: 校准数据集类型，可选 "pileval", "coco" 或 None (默认: "pileval")
        --n_samples: 校准样本数量 (默认: 128)
        --data_path: 数据路径
        --image_folder: 图像文件夹路径 (用于多模态数据)
        --interleave_format: 是否使用交错格式
        --few_shot_format: 是否使用少样本格式
        --text_data_path: 文本数据路径

        量化参数:
        --method: 量化方法，可选 "awq", "smoothquant", "mbq", "rtn" 或 None (默认: "awq")
        --w_bit: 权重量化位数 (默认: 8)
        --a_bit: 激活量化位数 (默认: 16)
        --w_group: 权重分组大小 (默认: 128)
        --alpha: 平滑参数 (默认: 0.5)
        --reweight: 是否启用重加权
        --distort: 是否启用失真
        --loss_mode: 损失函数模式，可选 "mae" 或 "mse" (默认: "mae")
        --scale_path: 缩放因子文件路径
        --run_process: 是否运行处理流程
        --pseudo_quant: 是否使用伪量化
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    # 基础模型参数
    parser.add_argument("--config", default="", help="Path to a yaml file specifying all eval arguments, will ignore cli arguments if specified")
    parser.add_argument("--model", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--model_args",
        default="",
        help="String arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu)",
    )
    # 校准数据参数
    parser.add_argument("--calib_data", default="pileval", choices=["pileval", "coco", None])
    parser.add_argument("--n_samples", default=128, type=int)
    parser.add_argument("--data_path", default="", type=str)
    parser.add_argument("--image_folder", default="", type=str)
    parser.add_argument("--interleave_format", action="store_true")
    parser.add_argument("--few_shot_format", action="store_true")
    parser.add_argument("--text_data_path", default="", type=str)

    # 量化参数
    parser.add_argument("--method", default="awq", choices=["awq", "smoothquant", "mbq", "rtn", None])
    parser.add_argument("--w_bit", default=8, type=int)
    parser.add_argument("--a_bit", default=16, type=int)
    parser.add_argument("--w_group", default=128, type=int)
    parser.add_argument("--alpha", default=0.5, type=int)
    parser.add_argument("--reweight", action="store_true")
    parser.add_argument("--distort", action="store_true")
    parser.add_argument("--loss_mode", default="mae", choices=["mae", "mse"])
    parser.add_argument("--scale_path", default=None, type=str)
    parser.add_argument("--run_process", action="store_true")
    parser.add_argument("--pseudo_quant", action="store_true")
    args = parser.parse_args()
    return args

def cli_quant_single(args: Union[argparse.Namespace, None] = None) -> None:
    """
    执行单次量化过程的核心函数

    该函数完成以下主要步骤：
    1. 加载指定的视觉-语言模型
    2. 预处理模型以适配量化
    3. 准备校准数据集
    4. 调用量化包装器执行量化

    参数:
        args (Union[argparse.Namespace, None]): 量化参数配置
    """
    # 在评估器外部加载MLLMs（多模态大语言模型）
    if args.model_args is None:
        args.model_args = ""

    # ==================新增的兼容代码====================
    # 解析 model_args 字符串，并处理 pretrained 参数
    model_args_dict = {}
    if args.model_args:
        for arg_pair in args.model_args.split(","):
            if "=" in arg_pair:
                key, value = arg_pair.split("=", 1)
                # 去除 pretrained 参数值两端的引号
                if key.strip() == "pretrained":
                    value = value.strip().strip('"')
                model_args_dict[key.strip()] = value.strip()
    
    # 重新构造 model_args 字符串
    reconstructed_model_args = ",".join([f"{k}={v}" for k, v in model_args_dict.items()])
    args.model_args = reconstructed_model_args
    # ====================新增的兼容代码=======================

    # 获取模型类并创建模型实例
    ModelClass = get_model(args.model)  # 加载模型类
    lm = ModelClass.create_from_arg_string(
        args.model_args,
        {
            "batch_size": args.batch_size,
            "device": args.device,
        },
    )

    # 预处理MLLM，使用 "lm._model" 获取fp16精度的模型
    Process_ModelClass = get_process_model(args.model)
    process_model = Process_ModelClass(lm._model,
                                       lm._tokenizer,
                                       lm.processor if hasattr(lm, 'processor') else None)

    # 生成校准tokens
    prompt_inputs = None
    prompt_kwargs = None

    # 根据指定的校准数据类型加载相应的数据集
    if args.calib_data == "pileval":
        # 加载文本校准数据集
        prompt_inputs, prompt_kwargs = get_calib_dataset(data_path=args.data_path, tokenizer=lm._tokenizer, n_samples=args.n_samples)
    elif args.calib_data == "coco":
        # 加载多模态校准数据集
        prompt_inputs, prompt_kwargs = get_multimodal_calib_dataset(data_path=args.data_path,
                                                                    image_folder=args.image_folder,
                                                                    model=process_model,
                                                                    n_samples=args.n_samples,
                                                                    few_shot_format=args.few_shot_format,
                                                                    interleave_format=args.interleave_format,
                                                                    text_data_path=args.text_data_path)

    # 包装量化模型并执行量化过程
    qwrapper(process_model, prompt_inputs, prompt_kwargs, args)

def cli_quant(args: Union[argparse.Namespace, None] = None) -> None:
    """
    量化过程的主CLI入口函数

    该函数处理命令行参数或配置文件，并为每个配置执行单次量化过程。
    支持通过YAML配置文件指定多个实验配置。

    参数:
        args (Union[argparse.Namespace, None]): 命令行参数对象。如果为None，则通过parse_quant_args()解析命令行参数。

    异常:
        ValueError: 当指定的配置文件不存在时抛出
    """
    if not args:
        # 如果没有提供参数，则解析命令行参数
        args = parse_quant_args()

    args_list = []
    if args.config:
        # 如果指定了配置文件，则从配置文件加载参数
        if not os.path.exists(args.config):
            raise ValueError(f"Config file does not exist: {args.config}")

        with open(args.config, "r") as file:
            config_args = yaml.safe_load(file)
        # 确保config_args是列表格式
        config_args = [config_args] if type(config_args) != list else config_args
        # 处理多个配置，为每个配置创建参数列表
        for config in config_args:
            # 复制基础参数
            args_copy = argparse.Namespace(**vars(args))
            # 用配置文件中的参数覆盖基础参数
            for key, value in config.items():
                setattr(args_copy, key, value)
            args_list.append(args_copy)
    else:
        # 如果没有指定配置文件，则直接使用传入的参数
        args_list.append(args)

    # 为每个参数配置执行单次量化过程
    for args in args_list:
        cli_quant_single(args)




if __name__ == "__main__":
    cli_quant()
