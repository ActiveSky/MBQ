# 读取一个pt文件并输出一些信息
import torch

# 读取pt文件
file_path = 'scale_cache/mbq/internvl2_8b_w3g128.pt'
data = torch.load(file_path)

# 保存到一个txt文件中
with open('output.txt', 'w') as f:
    f.write(str(data))
    f.write('\n')

print("save to output.txt")