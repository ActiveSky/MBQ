from modelscope.msdatasets import MsDataset

cache_dir = '/my-datasets'
ds =  MsDataset.load(
    'modelscope/coco_2014_caption',
    subset_name='coco_2014_caption',
    split='validation',
    cache_dir=cache_dir)

print(ds)