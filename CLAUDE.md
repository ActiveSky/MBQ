# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements Modality-Balanced Quantization (MBQ) for Large Vision-Language Models (VLMs). It supports quantization methods like MBQ, AWQ, SmoothQuant, and RTN for models such as InternVL2, LLaVA-OneVision, LLaVA-v1.5, Qwen2-VL, and VILA.

## Common Development Commands

### Installation
```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:thu-nics/MBQ.git

# Create conda environment
conda create -n qmllm python=3.10

# Install third-party packages
cd ./3rdparty/LLaVA-NeXT
pip install -e .

cd ./3rdparty/lmms-eval
pip install -e .

# Install main package
pip install -r requirements.txt
pip install -e .
```

### Running Quantization
```bash
# Using YAML config
python3 -W ignore main_quant.py --config configs/internvl2/MBQ_search/8b_weight_only.yaml

# Using command line arguments
python3 -W ignore main_quant.py \
    --model internvl2 \
    --model_args pretrained="OpenGVLab/InternVL2-8B" \
    --calib_data coco \
    --data_path "your/data/path/" \
    --image_folder "your/image/folder" \
    --n_samples 128 \
    --method mbq \
    --run_process \
    --w_bit 4 \
    --w_group 128 \
    --reweight \
    --loss_mode mae \
    --scale_path "scale_cache/mbq/internvl2_w4g128.pt"
```

### Running Evaluation
```bash
# Using YAML config
python3 -W ignore main.py --config configs/internvl2/Eval/eval.yaml

# Using command line arguments
python3 -W ignore main.py \
    --model internvl2 \
    --model_args pretrained="OpenGVLab/InternVL2-8B" \
    --tasks mmmu \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix mmmu \
    --method mbq \
    --pseudo_quant \
    --w_bit 4 \
    --w_group 128 \
    --output_path "your/output/path" \
    --scale_path "scale_cache/mbq/internvl2_w4g128.pt"
```

## Code Architecture

### Directory Structure
- `qmllm/`: Main package containing all quantization methods and utilities
  - `methods/`: Implementation of different quantization methods (mbq, awq, smoothquant, rtn)
  - `models/`: Model-specific implementations for different VLM architectures
  - `quantization/`: Core quantization utilities and wrappers
  - `calibration/`: Data loading and preprocessing for calibration
  - `datasets/`: Dataset handling utilities
- `configs/`: YAML configuration files organized by model type
- `3rdparty/`: Git submodules for external dependencies (LLaVA-NeXT, lmms-eval)

### Key Components
1. `main_quant.py`: Entry point for quantization search process
2. `main.py`: Entry point for model evaluation
3. `qmllm/quantization/quant_wrapper.py`: Main quantization wrapper that dispatches to specific methods
4. `qmllm/methods/*/entry.py`: Entry points for each quantization method
5. `qmllm/models/`: Model-specific loading and processing functions

### Quantization Flow
1. Model loading through `qmllm.models.get_process_model`
2. Calibration data preparation through `qmllm.calibration` modules
3. Quantization process execution via method-specific entry points
4. Scale/search results saved to disk
5. Pseudo-quantization applied for evaluation

### Supported Models
- InternVL2
- LLaVA-OneVision
- LLaVA-v1.5
- Qwen2-VL
- VILA

### Supported Methods
- MBQ (Modality-Balanced Quantization)
- AWQ (Activation-aware Weight Quantization)
- SmoothQuant
- RTN (Round-to-Nearest)