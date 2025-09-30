from datasets import load_dataset

cache_dir="/gz-data"
ds = load_dataset(
    "lmms-lab/COCO-Caption2017", 
    split="val",
  )



