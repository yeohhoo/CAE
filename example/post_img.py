from skimage.exposure import match_histograms
from PIL import Image
import numpy as np


def histogram_match_image(
    gen_img: Image.Image, ref_img: Image.Image
) -> Image.Image:
    gen = np.array(gen_img)
    ref = np.array(ref_img)
    matched = match_histograms(gen, ref, channel_axis=-1)
    return Image.fromarray(matched.astype(np.uint8))


# 真实图路径
ref = Image.open('/root/c_1206/test_flowers/o_flowers/anthurium/a_ori.png').convert('RGB')
# 对生成图后处理
gen_image_paths = ["/root/c_1206/test_flowers/t_flowers/anthurium/image_01989.png"]
for img_path in gen_image_paths:
    gen = Image.open(img_path).convert('RGB')
    new_img = histogram_match_image(gen, ref)
    new_img.save(img_path)  # 覆盖或另存
