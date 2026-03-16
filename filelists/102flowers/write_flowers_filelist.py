import numpy as np
import os
import json
import random

from collections import defaultdict
from PIL import Image
from os import listdir
from os.path import isfile, isdir, join


cwd = os.getcwd()
data_path = join(cwd, 'images')
savedir = './'
# dataset_list = ['base','val','novel']
dataset_list = ['base', 'val']

# if not os.path.exists(savedir):
#    os.makedirs(savedir)

folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
print(folder_list)
folder_list.sort()
label_dict = dict(zip(folder_list, range(0, len(folder_list))))

classfile_list_all = []

for i, folder in enumerate(folder_list):
    folder_path = join(data_path, folder)
    classfile_list_all.append(
        [
            join(folder_path, cf)
            for cf in listdir(folder_path)
            if (isfile(join(folder_path, cf)) and cf[0] != '.')
        ]
    )
    random.shuffle(classfile_list_all[i])

for dataset in dataset_list:
    file_list = []
    label_list = []
    for i, classfile_list in enumerate(classfile_list_all):
        if 'base' in dataset:
            # if (i%2 == 0):
            if i % 6 != 1:
                file_list = file_list + classfile_list
                label_list = (
                    label_list + np.repeat(i, len(classfile_list)).tolist()
                )
        if 'val' in dataset:
            # if (i%4 == 1):
            if i % 6 == 1:
                file_list = file_list + classfile_list
                label_list = (
                    label_list + np.repeat(i, len(classfile_list)).tolist()
                )
        # if 'novel' in dataset:
        #     if (i%4 == 3):
        #         file_list = file_list + classfile_list
        #         label_list = label_list + np.repeat(i, len(classfile_list)).tolist()
    print(f"number of {dataset} image:{np.unique(label_list).shape}")

    # if 'base' in dataset:
    #     for j in range(len(file_list)):
    #         print(f"base image:{file_list[j]}")
    #         img = Image.open(file_list[j])

    #         basepath = os.path.basename(file_list[j])

    #         label_dir = os.path.basename(os.path.dirname(file_list[j]))

    #         old_dir = os.path.dirname(os.path.dirname(os.path.dirname(file_list[j])))

    #         new_dir = os.path.join(old_dir, 'base', label_dir)
    #         if not os.path.exists(new_dir):
    #             os.makedirs(new_dir)

    #         img_path = os.path.join(new_dir, basepath)

    #         img.save(img_path)
    #     print(f"there are {j} base image")

    # if 'val' in dataset:
    #     for j in range(len(file_list)):
    #         print(f"val image:{file_list[j]}")
    #         img = Image.open(file_list[j])

    #         basepath = os.path.basename(file_list[j])

    #         label_dir = os.path.basename(os.path.dirname(file_list[j]))

    #         old_dir = os.path.dirname(os.path.dirname(os.path.dirname(file_list[j])))

    #         new_dir = os.path.join(old_dir, 'val', label_dir)
    #         if not os.path.exists(new_dir):
    #             os.makedirs(new_dir)

    #         img_path = os.path.join(new_dir, basepath)

    #         img.save(img_path)

    #     print(f"there are {j} val image")
    #     print(f"number of {dataset} image:{np.unique(label_list)}")

    fo = open(savedir + dataset + ".json", "w")
    if 'val' in dataset:
        fo_1 = open(savedir + dataset + ".list", "w")
        # fo_2 = open(savedir + dataset + "_2" + ".list", "w")
    if 'base' in dataset:
        fo_0 = open(savedir + dataset + "_0" + ".list", "w")
        fo_0_40 = open(savedir + dataset + "_0_40" + ".list", "w")

    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item for item in folder_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write('],')

    fo.write('"image_names": [')
    fo.writelines(['"%s",' % item for item in file_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write('],')

    # use for test

    if 'val' in dataset:
        num_val = 0
        num_val_1 = 0
        num_val_2 = 0
        val_list = []
        class_to_paths = defaultdict(list)

        for path in file_list:
            class_name = os.path.basename(os.path.dirname(path))
            class_to_paths[class_name].append(path)

        num_per_class = 40
        selected_images = {}

        for class_name, paths in class_to_paths.items():
            if len(paths) < num_per_class:
                print(
                    f"Warning: class '{class_name}' has only"
                    f" {len(paths)} images, less than {num_per_class}"
                )
                selected = paths  #
            else:
                selected = random.sample(paths, num_per_class)

            selected_images[class_name] = selected

        for cls, imgs in selected_images.items():
            print(f"{cls}:")
            for img in imgs:
                print(f"{img}")
                val_list.append(img)

        # fo_1.write('"image_names": [')
        # fo_2.write('"image_names": [')
        for i, item in enumerate(val_list):
            if i == 1:
                print(f"item={item}")
            num_val += 1
            # if i % 2 == 0:
            # num_val_1 += 1
            fo_1.writelines(['%s\n' % item])
            # fo_1.seek(0, os.SEEK_END)
            # fo_1.seek(fo_1.tell()-1, os.SEEK_SET)
            # else:
            #     num_val_2 += 1
            #     fo_2.writelines(['%s\n' % item ])
            # fo_2.seek(0, os.SEEK_END)
            # fo_2.seek(fo_2.tell()-1, os.SEEK_SET)

        # fo_1.write('],')
        # fo_2.write('],')

    if 'base' in dataset:
        num_base = 0
        for i, item in enumerate(file_list):
            num_base += 1
            fo_0.writelines(['%s\n' % item])

        num_base_40 = 0
        base_list = []
        class_to_paths_base = defaultdict(list)

        for path in file_list:
            class_name = os.path.basename(os.path.dirname(path))
            class_to_paths_base[class_name].append(path)

        num_per_class = 40
        selected_images = {}

        for class_name, paths in class_to_paths_base.items():
            if len(paths) < num_per_class:
                print(
                    f"Warning: class '{class_name}' has only"
                    f" {len(paths)} images, less than {num_per_class}"
                )
                selected = paths  #
            else:
                selected = random.sample(paths, num_per_class)

            selected_images[class_name] = selected

        for cls, imgs in selected_images.items():
            print(f"{cls}:")
            for img in imgs:
                print(f"{img}")
                base_list.append(img)

        # fo_1.write('"image_names": [')
        # fo_2.write('"image_names": [')
        for i, item in enumerate(base_list):
            if i == 1:
                print(f"item={item}")
            num_base_40 += 1
            # if i % 2 == 0:
            # num_val_1 += 1
            fo_0_40.writelines(['%s\n' % item])

    fo.write('"image_labels": [')
    fo.writelines(['%d,' % item for item in label_list])
    fo.seek(0, os.SEEK_END)
    fo.seek(fo.tell() - 1, os.SEEK_SET)
    fo.write(']}')

    fo.close()
    print("%s -OK" % dataset)

    if 'val' in dataset:
        fo_1.close()
        print("val_1-%s -OK" % dataset)
        # fo_2.close()
        # print("val_2-%s -OK" % dataset)
        print(f"the number of val:{num_val}")
        # print(f"the number of val_1:{num_val_1}")
        # print(f"the number of val_2:{num_val_2}")

    if 'base' in dataset:
        fo_0.close()
        print("base-%s -OK" % dataset)
        print(f"the number of base:{num_base}")
