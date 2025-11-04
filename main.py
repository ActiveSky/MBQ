import argparse
import datetime
import importlib
import json
import os
import sys
import traceback
import warnings
from functools import partial

import numpy as np
import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

import hashlib
from pathlib import Path
from typing import Union

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from loguru import logger as eval_logger

from lmms_eval import evaluator, utils
from lmms_eval.models import get_model
from lmms_eval.evaluator import request_caching_arg_to_dict
from lmms_eval.loggers import EvaluationTracker, WandbLogger
from lmms_eval.tasks import TaskManager
from lmms_eval.utils import (
    make_table,
    simple_parse_args_string,
)

from qmllm.quantization.quant_wrapper import qwrapper
from qmllm.models import get_process_model
from qmllm.calibration.pileval import get_calib_dataset
from qmllm.calibration.coco_vl import get_multimodal_calib_dataset


def _int_or_none_list_arg_type(min_len: int, max_len: int, defaults: str, value: str, split_char: str = ","):
    """
    解析一个逗号分隔的字符串，将其转换为整数或None的列表。
    如果只提供一个值，它将被复制以匹配最大长度。
    如果提供的项目数量不足，将使用默认值进行填充。

    Args:
        min_len (int): 列表的最小长度。
        max_len (int): 列表的最大长度。
        defaults (str): 默认值的逗号分隔字符串，用于填充缺失的值。
        value (str): 要解析的逗号分隔的字符串。
        split_char (str, optional): 分隔字符串的字符。默认为 ","。

    Returns:
        list: 包含整数或None的列表。

    Raises:
        argparse.ArgumentTypeError: 如果项目不是整数或None，或者项目数量不符合要求。
    """

    def parse_value(item):
        # 内部函数：解析单个值，可以是整数或"None"字符串
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} 不是一个整数或None")

    # 将输入字符串按分隔符拆分并解析每个值
    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        # 如果只提供一个值，将其复制以匹配最大长度，以便下游处理一致
        items = items * max_len
    elif num_items < min_len or num_items > max_len:
        # 检查项目数量是否在允许的范围内
        raise argparse.ArgumentTypeError(f"参数需要 {max_len} 个整数或None，以 '{split_char}' 分隔")
    elif num_items != max_len:
        # 如果项目数量不足最大长度，则使用默认值填充
        print(f"参数需要 {max_len} 个整数或None，以 '{split_char}' 分隔。 " "缺失的值将用默认值填充。")
        default_items = [parse_value(v) for v in defaults.split(split_char)]
        items.extend(default_items[num_items:])  # 使用缺失的默认值扩展列表

    return items

#没有被调用
def check_argument_types(parser: argparse.ArgumentParser):
    """
    检查所有命令行参数是否都指定了类型。
    如果任何非帮助参数没有指定类型，则会引发ValueError。

    Args:
        parser (argparse.ArgumentParser): 命令行参数解析器实例。

    Raises:
        ValueError: 如果发现有参数没有指定类型。
    """
    for action in parser._actions:
        # 遍历解析器中的所有动作（参数）
        if action.dest != "help" and not action.const:
            # 忽略 'help' 参数和常量参数
            if action.type is None:
                # 如果参数没有指定类型，则抛出错误
                raise ValueError(f"参数 '{action.dest}' 没有指定类型。")
            else:
                continue


def _handle_non_serializable(o):
    """
    处理JSON序列化过程中遇到的不可序列化对象。
    将numpy的int64/int32转换为Python的int，将set转换为list，其他不可序列化对象转换为字符串。

    Args:
        o (Any): 待处理的对象。

    Returns:
        Union[int, list, str]: 序列化后的对象。
    """
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        # 将numpy的整数类型转换为Python的int
        return int(o)
    elif isinstance(o, set):
        # 将set类型转换为list
        return list(o)
    else:
        # 其他不可序列化类型转换为字符串
        return str(o)


def parse_eval_args() -> argparse.Namespace:
    """
    解析命令行参数，用于配置模型评估。

    Returns:
        argparse.Namespace: 包含所有解析后的命令行参数的对象。
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config",
        default="",
        help="指定所有评估参数的 YAML 文件路径，如果指定，将忽略命令行参数。",
    )
    parser.add_argument("--model", default="hf", help="模型名称，例如 `hf`。")
    parser.add_argument(
        "--tasks",
        default=None,
        help="要获取完整的任务列表，请使用命令 `lmms-eval --tasks list`。",
    )
    parser.add_argument(
        "--model_args",
        default="",
        help="模型的字符串参数，例如 `pretrained=EleutherAI/pythia-160m,dtype=float32`。",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="few-shot 上下文中的示例数量。",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="可接受的值为 'auto'、'auto:N' 或 N，其中 N 是一个整数。默认为 1。",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="当 `--batch_size auto` 时尝试的最大批处理大小。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="要使用的设备（例如 cuda, cuda:0, cpu）。",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        metavar="= [dir/file.jsonl] [DIR]",
        help="保存结果指标的输出文件路径。如果路径是目录且 `log_samples` 为 True，则结果将保存到该目录。否则将使用父目录。",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="限制每个任务的示例数量。如果 <1，则限制为总示例数量的百分比。",
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="用于缓存模型响应的 sqlite 数据库文件路径。不缓存时为 `None`。",
    )
    parser.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        choices=["true", "refresh", "delete"],
        help="通过缓存数据集请求的构建来加速评估。不缓存时为 `None`。",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="是否运行任务测试套件的相关部分。",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="打印前几个文档的提示。",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        default=False,
        help="如果为 True，则输出所有模型输出和文档，用于每个样本的测量和事后分析。",
    )
    parser.add_argument(
        "--wandb_log_samples",
        action="store_true",
        default=False,
        help="如果为 True，则将所有模型输出和文档输出到 Weights and Biases，用于每个样本的测量和事后分析。",
    )
    parser.add_argument(
        "--log_samples_suffix",
        type=str,
        default="model_outputs",
        help="指定 log_samples 文件名的后缀。",
    )
    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        help="在提示中使用的系统指令。",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
        help="如果为 True，则将聊天模板应用于提示。",
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        default=False,
        help="如果为 True，则将 fewshot 作为多轮对话使用。",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="如果为 True，则在评估结束时显示所有任务的完整配置。",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="如果包含外部任务，需要包含的额外路径。",
    )
    parser.add_argument(
        "--gen_kwargs",
        default="",
        help=("用于 greedy_until 任务的模型生成字符串参数，" "例如 `temperature=0,top_k=0,top_p=0`。"),
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="任务未注册时记录错误。",
    )
    parser.add_argument(
        "--wandb_args",
        default="",
        help="传递给 wandb.init 的逗号分隔字符串参数，例如 `project=lmms-eval,job_type=eval`。",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Singapore",
        help="日期时间字符串的时区，例如 Asia/Singapore, America/New_York, America/Los_Angeles。您可以通过 `import pytz; print(pytz.common_timezones)` 查看完整列表。",
    )
    parser.add_argument(
        "--hf_hub_log_args",
        type=str,
        default="",
        help="传递给 Hugging Face Hub 日志函数的逗号分隔字符串参数，例如 `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`。",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="与 --log_samples 一起使用。只保存模型输出，不评估指标。",
    )
    default_seed_string = "0,1234,1234,1234"
    parser.add_argument(
        "--seed",
        type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),
        default=default_seed_string,  # for backward compatibility
        help=(
            "设置 python 的 random、numpy、torch 和 fewshot 采样的种子。\n"
            "接受一个逗号分隔的 4 个值列表，分别用于 python 的 random、numpy、torch 和 fewshot 采样种子，"
            "或者一个整数来为所有四个设置相同的种子。\n"
            f"这些值可以是整数或 'None'，表示不设置种子。默认值为 `{default_seed_string}` "
            "(为了向后兼容)。\n"
            "例如，`--seed 0,None,8,52` 设置 `random.seed(0)`、`torch.manual_seed(8)`，并将 fewshot 采样种子设置为 52。"
            "这里 numpy 的种子未设置，因为第二个值为 `None`。\n"
            "例如，`--seed 42` 将所有四个种子设置为 42。"
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="将 trust_remote_code 设置为 True，以执行代码从 Hub 创建 HF 数据集。",
    )

    # 校准参数
    parser.add_argument("--calib_data", default="pileval", choices=["pileval", "coco", None], help="校准数据集，可选 'pileval', 'coco' 或 None。")
    parser.add_argument("--data_path", default="", type=str, help="数据路径。")
    parser.add_argument("--image_folder", default="", type=str, help="图像文件夹路径。")
    parser.add_argument("--n_samples", default=128, type=int, help="校准样本数量。")
    parser.add_argument("--interleave_format", action="store_true", help="是否使用交错格式。")

    # TODO: 量化参数
    parser.add_argument("--method", default="awq", choices=["awq", "smoothquant", "mbq", "rtn", None], help="量化方法，可选 'awq', 'smoothquant', 'mbq', 'rtn' 或 None。")
    parser.add_argument("--w_bit", default=8, type=int, help="权重位宽。")
    parser.add_argument("--a_bit", default=16, type=int, help="激活位宽。")
    parser.add_argument("--w_group", default=128, type=int, help="权重分组大小。")
    parser.add_argument("--alpha", default=0.5, type=int, help="Alpha 参数。")
    parser.add_argument("--reweight", action="store_true", help="是否重新加权。")
    parser.add_argument("--distort", action="store_true", help="是否进行失真。")
    parser.add_argument("--loss_mode", default="mae", choices=["mae", "mse"], help="损失模式，可选 'mae' 或 'mse'。")
    parser.add_argument("--scale_path", default=None, type=str, help="量化比例因子路径。")
    parser.add_argument("--run_process", action="store_true", help="是否运行处理。")
    parser.add_argument("--pseudo_quant", action="store_true", help="是否进行伪量化。")
    args = parser.parse_args()
    return args

#没有被调用
def print_results(args, results):
    """
    打印评估结果的摘要。

    Args:
        args (argparse.Namespace): 命令行参数对象。
        results (dict): 评估结果字典。
    """
    # 打印模型、模型参数、生成参数、限制、few-shot 数量和批处理大小
    print(f"{args.model} ({args.model_args}),\ngen_kwargs: ({args.gen_kwargs}),\nlimit: {args.limit},\nnum_fewshot: {args.num_fewshot},\nbatch_size: {args.batch_size}")
    # 打印主要结果表格
    print(evaluator.make_table(results))
    if "groups" in results:
        # 如果结果中包含分组信息，则打印分组结果表格
        print(evaluator.make_table(results, "groups"))


def cli_evaluate_single(args: Union[argparse.Namespace, None] = None) -> None:
    """
    执行单个模型评估。
    此函数负责任务管理、模型加载、量化处理、评估执行和结果保存。

    Args:
        args (Union[argparse.Namespace, None], optional): 命令行参数对象。默认为 None。

    Returns:
        Tuple[dict, dict]: 评估结果和样本数据。
    """
    selected_task_list = args.tasks.split(",") if args.tasks else None # 将任务字符串按逗号分割成列表

    if args.include_path is not None:
        eval_logger.info(f"包含路径: {args.include_path}") # 记录包含路径信息
    # 初始化任务管理器
    task_manager = TaskManager(args.verbosity, include_path=args.include_path, model_name=args.model)

    # 使用输出路径和 HF token 更新评估跟踪器参数
    if args.output_path:
        args.hf_hub_log_args += f",output_path={args.output_path}" # 将输出路径添加到 HF Hub 日志参数中
    if os.environ.get("HF_TOKEN", None):
        args.hf_hub_log_args += f",token={os.environ.get('HF_TOKEN')}" # 将 HF Token 添加到 HF Hub 日志参数中

    evaluation_tracker_args = simple_parse_args_string(args.hf_hub_log_args) # 解析评估跟踪器参数字符串
    eval_logger.info(f"评估跟踪器参数: {evaluation_tracker_args}") # 记录评估跟踪器参数

    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args) # 初始化评估跟踪器

    if args.predict_only:
        args.log_samples = True # 如果只进行预测，则强制记录样本
    if (args.log_samples or args.predict_only) and not args.output_path:
        # 如果需要记录样本或只进行预测但未指定输出路径，则抛出错误
        raise ValueError("如果提供 --log_samples 或 --predict_only，请指定 --output_path")

    if args.fewshot_as_multiturn and args.apply_chat_template is False:
        # 如果 fewshot 作为多轮对话使用，则必须应用聊天模板
        raise ValueError("如果设置了 fewshot_as_multiturn，则 apply_chat_template 必须设置为 True。")

    if (args.num_fewshot is None or args.num_fewshot == 0) and args.fewshot_as_multiturn:
        # 如果 fewshot 作为多轮对话使用，则 fewshot 数量必须大于 0
        raise ValueError("如果设置了 fewshot_as_multiturn，则 num_fewshot 必须大于 0。")

    if args.include_path is not None:
        eval_logger.info(f"包含路径: {args.include_path}") # 再次记录包含路径信息

    if "push_samples_to_hub" in evaluation_tracker_args and not args.log_samples:
        eval_logger.warning("将样本推送到 Hub 需要设置 --log_samples。样本将不会被推送到 Hub。") # 警告信息

    if args.limit:
        eval_logger.warning(" --limit 仅用于测试。不应使用 LIMIT 计算真实指标。") # 警告信息

    if os.environ.get("LMMS_EVAL_PLUGINS", None):
        # 处理 LMMS_EVAL_PLUGINS 环境变量
        args.include_path = [args.include_path] if args.include_path else [] # 确保 include_path 是列表
        for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
            package_tasks_location = importlib.util.find_spec(f"{plugin}.tasks").submodule_search_locations[0] # 查找插件任务位置
            args.include_path.append(package_tasks_location) # 添加插件任务路径

    if args.tasks is None:
        eval_logger.error("需要指定要评估的任务。") # 错误信息
        sys.exit() # 退出程序
    elif args.tasks == "list":
        eval_logger.info("可用任务:\n - {}".format(f"\n - ".join(sorted(task_manager.list_all_tasks())))) # 列出所有可用任务
        sys.exit() # 退出程序
    elif args.tasks == "list_groups":
        eval_logger.info(task_manager.list_all_tasks(list_subtasks=False, list_tags=False)) # 列出所有任务组
        sys.exit() # 退出程序
    elif args.tasks == "list_tags":
        eval_logger.info(task_manager.list_all_tasks(list_groups=False, list_subtasks=False)) # 列出所有任务标签
        sys.exit() # 退出程序
    elif args.tasks == "list_subtasks":
        eval_logger.info(task_manager.list_all_tasks(list_groups=False, list_tags=False)) # 列出所有子任务
        sys.exit() # 退出程序
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = args.tasks.split(",") # 将任务列表字符串按逗号分割
            task_names = task_manager.match_tasks(task_list) # 匹配任务名称
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [task for task in task_list if task not in task_names and "*" not in task]  # 我们不希望在使用通配符 ("*") 任务名称时出现错误

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"未找到任务: {missing}\n" f"{utils.SPACING}尝试 `lm-eval --tasks list` 获取可用任务列表",
                )
                raise ValueError(
                    f"未找到任务: {missing}。尝试 `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` 列出所有可用任务分组名称；仅 (子)任务；标签；或以上所有，或传递 '--verbosity DEBUG' 以解决任务注册问题。"
                )

    eval_logger.info(f"选定任务: {task_names}") # 记录选定的任务
    request_caching_args = request_caching_arg_to_dict(cache_requests=args.cache_requests) # 获取请求缓存参数
    datetime_str = utils.get_datetime_str(timezone=args.timezone) # 获取当前日期时间字符串

    # 在评估器外部加载 MLLM 模型。
    if args.model_args is None:
        args.model_args = "" # 如果模型参数为 None，则设置为空字符串
    
    ModelClass = get_model(args.model) # 获取模型类
    lm = ModelClass.create_from_arg_string(
        args.model_args,
        {
            "batch_size": args.batch_size,
            "device": args.device,
        },
    )
    # 如果启用伪量化
    if args.pseudo_quant:
        print("伪量化中...") # 打印伪量化信息
        Process_ModelClass = get_process_model(args.model) # 获取处理模型类
        process_model = Process_ModelClass(lm._model, 
                                        lm._tokenizer, 
                                        lm.processor if hasattr(lm, 'processor') else None) # 初始化处理模型

        # 提示输入和提示参数
        prompt_inputs = None
        prompt_kwargs = None
        
        # 调用量化包装器
        qwrapper(process_model, prompt_inputs, prompt_kwargs, args)

    results = evaluator.simple_evaluate(
        model=args.model,
        lm=lm,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        evaluation_tracker=evaluation_tracker,
        system_instruction=args.system_instruction,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        verbosity=args.verbosity,
        predict_only=args.predict_only,
        random_seed=args.seed[0],
        numpy_random_seed=args.seed[1],
        torch_random_seed=args.seed[2],
        fewshot_random_seed=args.seed[3],
        cli_args=args,
        datetime_str=datetime_str,
        **request_caching_args,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples") # 如果记录样本，则从结果中弹出样本
        else:
            samples = None
        dumped = json.dumps(results, indent=4, default=_handle_non_serializable) # 将结果转换为 JSON 字符串
        if args.show_config:
            print(dumped) # 如果显示配置，则打印 JSON 结果

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"])) # 获取批处理大小

        evaluation_tracker.save_results_aggregated(results=results, samples=samples if args.log_samples else None, datetime_str=datetime_str) # 保存聚合结果

        if args.log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name]) # 保存每个任务的样本结果

        if evaluation_tracker.push_results_to_hub or evaluation_tracker.push_samples_to_hub:
            evaluation_tracker.recreate_metadata_card() # 如果需要推送到 Hub，则重新创建元数据卡

        return results, samples # 返回结果和样本
    return None, None # 如果结果为 None，则返回 None, None



def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    """
    命令行界面评估函数。这是整个评估流程的入口点。
    它解析命令行参数，初始化日志和加速器，处理多个配置，并调用 cli_evaluate_single 进行实际评估。

    Args:
        args (Union[argparse.Namespace, None], optional): 命令行参数对象。如果为 None，则会调用 parse_eval_args() 进行解析。
                                                          默认为 None。
    """
    if not args:
        # 如果没有提供参数，则解析命令行参数
        args = parse_eval_args()

    # 检查解析后是否未传递任何参数
    if len(sys.argv) == 1:
        print("┌───────────────────────────────────────────────────────────────────────────────┐")
        print("│ 请提供参数来评估模型。例如：                                                  │")
        print("│ `lmms-eval --model llava --model_path liuhaotian/llava-v1.6-7b --tasks okvqa` │")
        print("│ 使用 `lmms-eval --help` 获取更多信息。                                        │")
        print("└───────────────────────────────────────────────────────────────────────────────┘")
        sys.exit(1) # 如果没有参数，则退出程序

    if args.wandb_args:
        # 如果指定了 Weights & Biases (wandb) 参数
        if "name" not in args.wandb_args:
            # 如果 wandb 参数中没有指定名称，则生成一个名称
            name = f"{args.model}_{args.model_args}_{utils.get_datetime_str(timezone=args.timezone)}"
            name = utils.sanitize_long_string(name)
            args.wandb_args += f",name={name}"
        # 初始化 WandbLogger
        wandb_logger = WandbLogger(**simple_parse_args_string(args.wandb_args))

    # 重置日志记录器
    eval_logger.remove() # 移除所有现有的日志处理器
    eval_logger.add(sys.stdout, colorize=True, level=args.verbosity) # 添加新的日志处理器，输出到标准输出，带颜色，并设置日志级别
    eval_logger.info(f"日志级别设置为 {args.verbosity}") # 记录当前日志级别
    os.environ["VERBOSITY"] = args.verbosity # 设置环境变量 VERBOSITY
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # 禁用 tokenizers 的并行化，避免潜在的死锁或性能问题

    args_list = [] # 用于存储所有配置的参数列表
    results_list = [] # 用于存储所有配置的评估结果列表
    if args.config:
        # 如果指定了配置文件
        if not os.path.exists(args.config):
            # 检查配置文件是否存在
            raise ValueError(f"配置文件不存在: {args.config}")

        with open(args.config, "r") as file:
            # 读取 YAML 配置文件
            config_args = yaml.safe_load(file)
        # 如果配置文件中包含多个配置，将其转换为列表
        config_args = [config_args] if type(config_args) != list else config_args
        # 处理多个配置，为每个配置创建一个独立的参数对象
        for config in config_args:
            args_copy = argparse.Namespace(**vars(args)) # 复制当前参数
            for key, value in config.items():
                setattr(args_copy, key, value) # 用配置文件中的值覆盖或添加参数
            args_list.append(args_copy) # 将处理后的参数添加到列表中
    else:
        args_list.append(args) # 如果没有配置文件，则只使用当前解析的参数

    # 初始化 Accelerator，用于分布式训练/评估
    kwargs_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60000)) # 设置进程组初始化超时时间
    accelerator = Accelerator(kwargs_handlers=[kwargs_handler]) # 创建 Accelerator 实例
    if accelerator.is_main_process:
        # 判断当前是否为主进程
        is_main_process = True
    else:
        is_main_process = False

    for args in args_list:
        # 遍历所有参数配置进行评估
        try:
            # if is_main_process and args.wandb_args:  # 考虑到我们应该只初始化 wandb 一次，而不是多个 rank，以避免网络流量和不必要的行为。
            #     wandb_logger = WandbLogger()

            results, samples = cli_evaluate_single(args) # 执行单个评估
            results_list.append(results) # 将结果添加到结果列表中

            accelerator.wait_for_everyone() # 等待所有进程完成
            if is_main_process and args.wandb_args:
                # 如果是主进程且指定了 wandb 参数
                try:
                    wandb_logger.post_init(results) # wandb 后初始化
                    wandb_logger.log_eval_result() # 记录评估结果
                    if args.wandb_log_samples and samples is not None:
                        wandb_logger.log_eval_samples(samples) # 记录样本
                except Exception as e:
                    eval_logger.info(f"记录到 Weights and Biases 失败，原因: {e}") # 记录 wandb 记录失败信息
                # wandb_logger.finish()

        except Exception as e:
            # 捕获评估过程中发生的异常
            if args.verbosity == "DEBUG":
                raise e # 如果日志级别为 DEBUG，则重新抛出异常
            else:
                traceback.print_exc() # 打印异常堆栈信息
                eval_logger.error(f"评估过程中发生错误: {e}。请设置 `--verbosity=DEBUG` 以获取更多信息。") # 记录错误信息
                results_list.append(None) # 将 None 添加到结果列表，表示评估失败

    for args, results in zip(args_list, results_list):
        # 遍历所有参数和结果列表，打印评估结果
        # 如果进程不是主进程 (rank 0)，cli_evaluate 将返回 None
        if results is not None:
            print(f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, " f"batch_size: {args.batch_size}")
            print(make_table(results)) # 打印结果表格
            if "groups" in results:
                print(make_table(results, "groups")) # 如果有分组结果，则打印分组表格

    if args.wandb_args:
        wandb_logger.run.finish() # 完成 wandb 运行


if __name__ == "__main__":
    # 当脚本作为主程序运行时，调用 cli_evaluate 函数
    cli_evaluate()
