import os
import csv
import lpips
import torch
import open_clip
import shutil
import tempfile
import numpy as np

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from itertools import combinations
from pytorch_fid import fid_score
from semantic_directions import flower_classes


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 图像预处理（LPIPS & CLIP 共用）
transform_lpips = transforms.Compose(
    [transforms.Resize((256, 256)), transforms.ToTensor()]
)

transform_clip = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)


def load_images(folder, transform):
    images = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(('.jpg', '.png', '.jpeg')):
            img = Image.open(os.path.join(folder, f)).convert('RGB')
            images.append(transform(img).unsqueeze(0))  # [1, C, H, W]
    return torch.cat(images, dim=0) if images else None


# def compute_lpips_score(images, model):
#     total = 0.0
#     count = 0
#     for i, j in combinations(range(len(images)), 2):
#         d = model(images[i].unsqueeze(0), images[j].unsqueeze(0))
#         total += d.item()
#         count += 1
#     return total / count if count > 0 else 0.0


def compute_lpips_score(images, model, batch_size=512):
    total = 0.0
    count = 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()

    with torch.no_grad():
        pairs = list(combinations(range(len(images)), 2))

        for k in range(0, len(pairs), batch_size):
            batch = pairs[k : k + batch_size]

            imgs1 = torch.stack([images[i] for i, j in batch]).to(device)
            imgs2 = torch.stack([images[j] for i, j in batch]).to(device)

            dists = model(imgs1, imgs2, normalize=True)  # normalize视具体需求
            total += dists.sum().item()
            count += dists.shape[0]

    return total / count if count > 0 else 0.0


# def compute_clip_score(images, text_embed, model):
#     with torch.no_grad():
#         image_embed = model.encode_image(images)
#         image_embed = image_embed / image_embed.norm(dim=-1, keepdim=True)
#         text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
#         similarity = (image_embed @ text_embed.T).squeeze(-1)  # [N]
#         return similarity.mean().item()


def compute_clip_score(
    images, latent_labels, model, tokenizer, device, batch_size=2  # 32
):
    image_feats = []

    with torch.no_grad():
        for i in range(0, images.shape[0], batch_size):
            batch_imgs = images[i : i + batch_size]
            feats = model.encode_image(batch_imgs)
            feats /= feats.norm(dim=-1, keepdim=True)
            image_feats.append(feats.cpu())  # 可转存CPU，降低显存占用

    image_feats = torch.cat(image_feats, dim=0)  # (N, D)

    # pre-code text
    class_to_text_feat = {}
    for cls_idx in set(latent_labels):
        prompt = f"a photo of a {flower_classes[int(cls_idx)]}"
        tokens = tokenizer([prompt]).to(device)
        with torch.no_grad():
            text_feat = model.encode_text(tokens)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        class_to_text_feat[int(cls_idx)] = text_feat.cpu()

    # # calculate similarity
    # clip_scores = []
    # with torch.no_grad():
    #     for i, lat in enumerate(latent_labels):
    #         img_feat = image_feats[i].unsqueeze(0)  # (1, D)
    #         text_feat = class_to_text_feat[int(lat)]
    #         sim = (img_feat @ text_feat.T).item()
    #         clip_scores.append(sim)

    # return sum(clip_scores) / len(clip_scores)
    # Calculate similarity between image features and text features
    clip_scores = []
    with torch.no_grad():
        for i, lat in enumerate(latent_labels):
            img_feat = image_feats[i].unsqueeze(
                0
            )  # (1, D) - Add an extra dimension for batch size 1
            similarities = []

            # Calculate similarity with all class text features
            for class_idx, text_feat in class_to_text_feat.items():
                sim = (img_feat @ text_feat.T).item()  # Cosine similarity
                similarities.append(sim)

            # Choose the class with the highest similarity
            best_class_idx = similarities.index(max(similarities))
            best_similarity = max(similarities)

            clip_scores.append(
                (best_class_idx, best_similarity)
            )  # Store the predicted class and its score

    # Return the average similarity score
    avg_clip_score = sum([score[1] for score in clip_scores]) / len(
        clip_scores
    )
    return clip_scores, avg_clip_score


def resize_images(input_dir, output_dir, size=(299, 299)):
    os.makedirs(output_dir, exist_ok=True)
    transform = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ]
    )
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('png', 'jpg', 'jpeg', 'bmp')):
            img_path = os.path.join(input_dir, filename)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            # Save back to disk as .png
            save_path = os.path.join(output_dir, filename)
            transforms.ToPILImage()(img).save(save_path)


def duplicate_single_image(src_path, target_dir, n=128):
    os.makedirs(target_dir, exist_ok=True)
    img = Image.open(src_path).convert("RGB")
    for i in range(n):
        img.save(os.path.join(target_dir, f"real_{i}.png"))


def compute_fid_with_resize(
    path1, path2, image_size=(299, 299), device='cuda'
):
    # Create temp dirs for resized images
    tmp_dir1 = tempfile.mkdtemp()
    tmp_dir2 = tempfile.mkdtemp()

    try:
        print(f"Resizing images to {image_size}...")
        resize_images(path1, tmp_dir1, image_size)
        resize_images(path2, tmp_dir2, image_size)

        print("Computing FID...")
        fid_value = fid_score.calculate_fid_given_paths(
            [tmp_dir1, tmp_dir2],
            batch_size=256,  # 50
            device=device,
            dims=2048,
        )
        print(f"\nFID: {fid_value:.4f}")
        return fid_value

    finally:
        shutil.rmtree(tmp_dir1)
        shutil.rmtree(tmp_dir2)


def evaluate_all(
    real_root, fake_root, fake_label, output_csv="eval_result.csv"
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lpips_model = lpips.LPIPS(net='alex').to(device)

    # 加载 CLIP 模型
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # classes = sorted(os.listdir(real_root))
    results = []

    real_dir = real_root
    fake_dir = fake_root

    real_imgs = [
        f for f in os.listdir(real_dir) if f.lower().endswith(('jpg', 'png'))
    ]

    if len(real_imgs) == 1:
        tmp_real_dir = tempfile.mkdtemp()
        duplicate_single_image(
            os.path.join(real_dir, real_imgs[0]), tmp_real_dir, n=8
        )
        real_dir_for_fid = tmp_real_dir
    else:
        real_dir_for_fid = real_dir
    # Compute FID
    try:
        fid = fid_score.calculate_fid_given_paths(
            [real_dir, fake_dir], batch_size=512, device=device, dims=2048
        )
        # fid = compute_fid_with_resize(
        #     real_dir_for_fid,
        #     fake_dir,
        #     image_size=(256, 256),  # (256, 256)
        #     device=device,
        # )
        print(f"fid={fid}")
    except Exception as e:
        print(f"[FID] Failed for: {e}")
        fid = -1

    # Load fake images for LPIPS & CLIP
    try:
        fake_imgs_lpips = load_images(fake_dir, transform_lpips).to(device)
        fake_imgs_clip = load_images(fake_dir, transform_clip).to(device)
    except:
        print(f"Failed to load images")

    # # Compute LPIPS
    # try:
    #     lpips_score = compute_lpips_score(fake_imgs_lpips, lpips_model)
    #     print(f"lpips_score={lpips_score}")
    # except:
    #     lpips_score = -1

    # Compute CLIP Score
    try:
        # 采用提示模板提升语义一致性评估
        # load latent label .npy
        if os.path.exists(fake_label):
            print(f'Load latent codes from {fake_label}')
            latent_labels = np.load(fake_label)  # (1312,)
        else:
            raise ValueError('No latent_label_path file')

        clip_scores, avg_clip_score = compute_clip_score(
            fake_imgs_clip, latent_labels, model, tokenizer, device
        )

    except:
        clip_scores = -1

    # 输出每个图像的预测类别及其得分
    for idx, (best_class_idx, score) in enumerate(clip_scores):
        print(
            f"Image {idx}: Predicted class - {flower_classes[best_class_idx]}, CLIP Score: {score:.4f}"
        )

    # 输出所有图像的平均CLIP得分
    print(f"Average CLIP Score: {avg_clip_score:.4f}")
    # results.append(
    #     ("Metric", round(fid, 3), round(lpips_score, 4), round(clip_score, 4))
    # )
    # print(f"results={results}")

    # # Save to CSV
    # with open(output_csv, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["Class", "FID (↓)", "LPIPS (↑)", "CLIP-Score (↑)"])
    #     writer.writerows(results)

    # print(f"\n Done. Results saved to {output_csv}")


if __name__ == "__main__":
    real_folder = input("Path to real images folder: ").strip()
    fake_folder = input("Path to generated images folder: ").strip()
    fake_label = input("Path to generated images label .npy: ").strip()

    evaluate_all(real_folder, fake_folder, fake_label)
    # /root/c_1206/CAE/example/unseen_imgs
    # /root/c_1206/CAE/work_dirs/Manipulate/102flower/20250606_0720/images
