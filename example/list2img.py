import os
import shutil

flower_classes = [
    "pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea",
    "english marigold", "tiger lily", "moon orchid", "bird of paradise",
    "monkshood", "globe thistle", "snapdragon", "colt's foot",
    "king protea", "spear thistle", "yellow iris", "globe-flower",
    "purple coneflower", "peruvian lily", "balloon flower", "giant white arum lily",
    "fire lily", "pincushion flower", "fritillary", "red ginger",
    "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian",
    "artichoke", "sweet william", "carnation", "garden phlox",
    "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya",
    "cape flower", "great masterwort", "siam tulip", "lenten rose",
    "barbeton daisy", "daffodil", "sword lily", "poinsettia",
    "bolero deep blue", "wallflower", "marigold", "buttercup",
    "oxeye daisy", "common dandelion", "petunia", "wild pansy",
    "primula", "sunflower", "pelargonium", "bishop of llandaff",
    "gaura", "geranium", "orange dahlia", "pink-yellow dahlia",
    "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush",
    "californian poppy", "osteospermum", "spring crocus", "bearded iris",
    "windflower", "tree poppy", "gazania", "azalea",
    "water lily", "rose", "thorn apple", "morning glory",
    "passion flower", "lotus", "toad lily", "anthurium",
    "frangipani", "clematis", "hibiscus", "columbine",
    "desert-rose", "tree mallow", "magnolia", "cyclamen ",
    "watercress", "canna lily", "hippeastrum", "bee balm",
    "ball moss", "foxglove", "bougainvillea", "camellia",
    "mallow", "mexican petunia", "bromelia", "blanket flower",
    "trumpet creeper", "blackberry lily"
]  #"common tulip", "wild rose",


test_list = "/root/c_1206/CAE/example/test.list"

print('Loading image list.')
image_list = []
label_list = []
with open(test_list, 'r') as f:
    for line in f:
        image_list.append(line.strip())
        label_list.append(image_list[-1].split('/')[-2])

unique_labels = list(set(label_list))

src_dir = "/root/c_1206/CAE/filelists/102flowers/images"
dst_dir = "/root/c_1206/CAE/example/unseen_imgs"

os.makedirs(dst_dir, exist_ok=True)

for old_name in unique_labels:
    src_path = os.path.join(src_dir, old_name)
    new_name = flower_classes[int(old_name)]
    dst_path = os.path.join(dst_dir, new_name)

    if os.path.exists(src_path):
        try:
            shutil.copytree(src_path, dst_path)
            print(f"✅ 成功复制 {old_name} -> {new_name}")
        except Exception as e:
            print(f"❌ 复制失败 {old_name}: {e}")
    else:
        print(f"⚠️ 源文件夹 {old_name} 不存在，跳过")

print("🎉 所有文件夹复制完成！")