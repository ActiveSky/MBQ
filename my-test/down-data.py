
import json
import os
import requests
from urllib.parse import urlparse

def download_images_from_json(input_file, output_directory="downloaded_images"):
    """
    读取JSON文件，提取图片路径，构造完整的URL，下载图片到指定目录，
    但不修改原始JSON文件，也不生成新的处理后的JSON文件。

    Args:
        input_file (str): 包含图片信息的JSON文件名。
        output_directory (str): 下载图片保存的目录。
    """

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Starting image download process from '{input_file}'...")

    for i, entry in enumerate(data):
        print(f"\nProcessing entry {i+1}/{len(data)}...")
        original_image_path = entry.get("image")

        if original_image_path:
            # 构造新的URL
            # 假设原始路径是 "coco/train2017/000000000009.jpg"
            # 目标URL是 "http://images.cocodataset.org/train2017/000000000009.jpg"
            # 我们需要把 "coco/" 移除掉，然后拼接 baseURL
            parts = original_image_path.split('/', 1) # 分割一次，得到 "coco" 和 "train2017/000000000009.jpg"
            if len(parts) > 1:
                relative_path = parts[1]
            else:
                relative_path = original_image_path # 如果没有 "coco/" 前缀，直接使用原始路径

            base_url = "http://images.cocodataset.org/"
            new_image_url = base_url + relative_path

            # 下载图片
            try:
                print(f"Constructed URL for download: {new_image_url}")
                response = requests.get(new_image_url, stream=True, timeout=10) # 增加超时时间
                response.raise_for_status()  # 检查请求是否成功

                # 提取图片文件名 (例如 "000000000009.jpg")
                image_filename = os.path.basename(original_image_path)
                save_path = os.path.join(output_directory, image_filename)

                with open(save_path, 'wb') as img_file:
                    for chunk in response.iter_content(chunk_size=8192):
                        img_file.write(chunk)
                print(f"Successfully downloaded '{image_filename}' to '{output_directory}'")

            except requests.exceptions.Timeout:
                print(f"Error: Connection timed out while downloading {new_image_url}")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {new_image_url}: {e}")
            except Exception as e:
                print(f"An unexpected error occurred while processing {new_image_url}: {e}")
        else:
            print(f"Skipping entry {i+1} due to missing 'image' field.")

    print("\nImage download process completed.")


if __name__ == "__main__":
    input_filename = "my-datasets/coco/data.json"  # 你的输入JSON文件名
    download_folder = "my-datasets/coco/train2017"  # 图片下载保存的目录




    download_images_from_json(input_filename, download_folder)
