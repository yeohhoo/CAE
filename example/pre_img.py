from datasets.datamgr import TransformLoader
from utils.visualizer import save_image, load_image
from utils import common

def preprocess(image, image_size=256):
    """Preprocesses a single image.
    This function use the same transformation as training.
    """
    trans_loader = TransformLoader(image_size)
    transform = trans_loader.get_composed_transform()
    img = transform(image)  # shape with [3, 256, 256]
    return img

output_dir = "/root/c_1206"
image_path = "/root/c_1206/CAE/example/unseen_imgs/anthurium/image_01989.jpg"
image = load_image(image_path)
image = preprocess(image)
save_image(f'{output_dir}/a_ori.png', common.tensor2im(image))


