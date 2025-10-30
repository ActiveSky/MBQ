
## 数据处理过程
在 `qmllm/calibration/coco_vl.py` 文件中，`get_multimodal_calib_dataset` 函数在图像处理过程中调用了 `model` 对象的以下方法：

1.  `model.preprocess_data(images, data_item)`: 这个方法用于对加载的图像和对应的数据项进行预处理。
2.  `model.data_collator(data_list)`: 这个方法用于将预处理后的数据列表进行整理，通常是为了批处理。
3.  `model.few_shot_data_samples(examples)`: 如果 `few_shot_format` 参数为 `True`，则会调用此方法来处理 few-shot 格式的数据样本。
4.  `model.tokenizer.encode(line)`: 如果 `interleave_format` 参数为 `True`，并且需要处理纯文本数据，会通过 `model.tokenizer` 属性调用 `encode` 方法对文本行进行编码。
5.  `model.interleave_data_samples(examples, pure_text=pure_text)`: 如果 `interleave_format` 参数为 `True`，则会调用此方法将图像数据与纯文本数据进行交错处理。
6.  `model.generate_input(examples)`: 最后，这个方法用于根据整理好的 `examples` 生成模型所需的最终输入格式（`prompt_inputs` 和 `prompt_kwargs`）。

## 一次运行过程
```bash
(qmllm) ➜  MBQ git:(main) python3 -W ignore main_quant.py \
    --config configs/internvl2/MBQ_search/8b_weight_only.yaml
petrel_client is not installed. If you read data locally instead of from ceph, ignore it.
OpenCLIP not installed
FlashAttention2 is not installed.
InternLM2ForCausalLM has generative capabilities, as `prepare_inputs_for_generation` is explicitly overwritten. However, it doesn't directly inherit from `GenerationMixin`. From 👉v4.50👈 onwards, `PreTrainedModel` will NOT inherit from `GenerationMixin`, and this model will lose the ability to call `generate` and other related functions.
  - If you're using `trust_remote_code=True`, you can get rid of this warning by loading the model with an auto class. See https://huggingface.co/docs/transformers/en/model_doc/auto#auto-classes
  - If you are the owner of the model architecture code, please modify your model class such that it inherits from `GenerationMixin` (after `PreTrainedModel`, otherwise you'll get an exception).
  - If you are not the owner of the model architecture class, please contact the model code owner to update it.
Warning: Flash attention is not available, using eager attention instead.
Loading checkpoint shards: 100%|█████████████████████████████████████████████████████████████████████████████| 4/4 [00:29<00:00,  7.50s/it]
Save gradient...
Running gradient calculation...: 100%|█████████████████████████████████████████████████████████████████████| 16/16 [00:06<00:00,  2.61it/s]
Running MBQ...:  53%|█████████████████████████████████████████████▋                                        | 17/32 [00:58<0Running MBQ...:  56%|████████████████████████████████████████████████▍                                     | 18/32 [01:01<0Running MBQ...:  59%|███████████████████████████████████████████████████                                   | 19/32 [01:05<0Running MBQ...:  62%|█████████████████████████████████████████████████████▊                                | 20/32 [01:08<0Running MBQ...:  66%|████████████████████████████████████████████████████████▍                             | 21/32 [01:11<0Running MBQ...:  69%|███████████████████████████████████████████████████████████▏                          | 22/32 [01:15<0Running MBQ...:  72%|█████████████████████████████████████████████████████████████▊                        | 23/32 [01:18<0Running MBQ...:  75%|████████████████████████████████████████████████████████████████▌                     | 24/32 [01:22<0Running MBQ...:  78%|███████████████████████████████████████████████████████████████████▏                  | 25/32 [01:25<0Running MBQ...:  81%|█████████████████████████████████████████████████████████████████████▉                | 26/32 [01:29<0Running MBQ...:  84%|████████████████████████████████████████████████████████████████████████▌             | 27/32 [01:32<0Running MBQ...:  88%|███████████████████████████████████████████████████████████████████████████▎          | 28/32 [01:35<0Running MBQ...:  91%|█████████████████████████████████████████████████████████████████████████████▉        | 29/32 [01:39<0Running MBQ...:  94%|████████████████████████████████████████████████████████████████████████████████▋     | 30/32 [01:42<0Running MBQ...:  97%|███████████████████████████████████████████████████████████████████████████████████▎  | 31/32 [01:45<0Running MBQ...: 100%|██████████████████████████████████████████████████████████████████████████████████████| 32/32 [01:49<0Running MBQ...: 100%|██████████████████████████████████████████████████████████████████████████████████████| 32/32 [01:49<00:00,  3.42s/it]
MBQ results saved at scale_cache/mbq/internvl2_8b_w3g128.pt

```

