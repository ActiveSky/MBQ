import os
import json
import torch
import numpy as np

from PIL import Image
from datasets import load_dataset

def load_image(image_path):
    """
    加载图像文件。

    Args:
        image_path (str): 图像文件的路径。

    Returns:
        PIL.Image.Image: 加载并转换为RGB格式的图像。
    """
    # 使用 PIL 加载图像，目前不支持 tcs_loader
    return Image.open(image_path).convert('RGB')


def get_multimodal_calib_dataset(
    data_path,
    image_folder,
    model,
    n_samples=128,
    few_shot_format=False,
    interleave_format=False,
    text_data_path=None,
    shuffle=True, 
):
    """
    获取多模态校准数据集。

    Args:
        data_path (str): 数据集文件的路径，支持 .jsonl 或 .json 格式。
        image_folder (str): 图像文件所在的根目录。
        model: 用于数据预处理、数据整理和输入生成的模型对象。
        n_samples (int, optional): 要加载的样本数量。默认为 128。
        few_shot_format (bool, optional): 是否使用 few-shot 格式。默认为 False。
        interleave_format (bool, optional): 是否使用交错格式。默认为 False。
        text_data_path (str, optional): 纯文本数据集的路径，用于交错格式。如果为 None，则加载默认数据集。
                                        默认为 None。
        shuffle (bool, optional): 是否打乱数据集。默认为 True。

    Returns:
        tuple: 包含 prompt_inputs 和 prompt_kwargs 的元组，用于模型生成输入。

    Raises:
        ValueError: 如果文件类型不支持，或者同时指定了 few_shot_format 和 interleave_format。
    """
    # 根据文件类型加载数据集
    if data_path.endswith(".jsonl"):
        dataset = []
        with open(data_path, "r") as json_file:
            for line in json_file:
                dataset.append(json.loads(line.strip()))
    elif data_path.endswith(".json"):
        with open(data_path, "r") as json_file:
            dataset = json.load(json_file)
    else:
        # 如果文件类型不支持，则抛出错误
        raise ValueError(f"Unsupported file type: {data_path}")
    
    # 如果需要打乱数据集
    if shuffle:
        rng = np.random.default_rng(seed=42) # 使用固定种子以保证可复现性
        rng.shuffle(dataset) # 打乱数据集

    data_list = []
    # 遍历指定数量的样本
    for i in range(n_samples):
        i = i % len(dataset) # 循环使用数据集，以防 n_samples 大于数据集大小
        data_item = dataset[i]
        # 检查数据项中是否存在图像
        if 'image' in data_item and len(data_item['image']) != 0:
            if type(data_item['image']) == list:
                images = []
                # 处理多张图像
                for image_path in data_item['image']:
                    # 合并图像路径
                    full_image_path = os.path.join(image_folder, image_path)
                    image = load_image(full_image_path)
                    images.append(image)
            else:
                images = []
                # 处理单张图像
                image_path = data_item['image']
                full_image_path = os.path.join(image_folder, image_path)
                image = load_image(full_image_path)
                images.append(image)
        else:
            images = None # 如果没有图像，则设置为 None
        
        # 使用模型预处理数据
        data_dict = model.preprocess_data(images, data_item)
        data_list.append(data_dict)

    # 使用模型的数据整理器处理数据列表
    examples = model.data_collator(data_list)
    
    # 检查 few_shot_format 和 interleave_format 是否同时指定
    if few_shot_format and interleave_format:
        raise ValueError('You cannot specify both few_shot_format and interleave_format at the same time!')

    # 如果使用 few-shot 格式
    if few_shot_format:
        examples = model.few_shot_data_samples(examples)
    
    # 如果使用交错格式
    if interleave_format:
        if not text_data_path:
            # 如果没有指定文本数据路径，则加载默认数据集
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        else:
            # TODO: 对于其他数据集，可能不应该指定 split 参数
            dataset = load_dataset(data_path, split="validation")
        if shuffle:
            dataset = dataset.shuffle(seed=42) # 打乱文本数据集
        samples = []
        n_run = 0
        # 遍历文本数据集，提取纯文本样本
        for data in dataset:
            line = data["text"].strip()
            line_encoded = model.tokenizer.encode(line) # 对文本进行编码
            
            if len(line_encoded) > 512:
                sample = torch.tensor(line_encoded[:512]) # 截断过长的文本
                samples.append(sample)
                n_run += 1
                
            if n_run == 128: # 收集到足够样本后停止
                break
        pure_text = samples # 纯文本样本列表

        # 使用模型将图像数据与纯文本数据交错
        examples = model.interleave_data_samples(examples, pure_text=pure_text)

    # 使用模型生成输入
    prompt_inputs, prompt_kwargs = model.generate_input(examples)

    return prompt_inputs, prompt_kwargs # 返回生成的输入
