import os
import cv2
import numpy as np
import torch
from tqdm import tqdm

def pre_processing(img, T_H=64, T_W=64):
    if img.sum() <= 10000:
        return None

    y = img.sum(axis=1)
    y_top = (y != 0).argmax()
    y_btm = (y != 0).cumsum().argmax()
    img = img[y_top:y_btm + 1, :]

    r = img.shape[1] / img.shape[0]
    target_w = int(T_H * r)
    img = cv2.resize(img, (target_w, T_H), interpolation=cv2.INTER_CUBIC)

    sum_total = img.sum()
    sum_col = img.sum(axis=0).cumsum()
    x_center = np.searchsorted(sum_col, sum_total // 2)

    h_T_W = T_W // 2
    left = x_center - h_T_W
    right = x_center + h_T_W

    if left < 0 or right >= img.shape[1]:
        pad = np.zeros((img.shape[0], h_T_W), dtype=img.dtype)
        img = np.concatenate([pad, img, pad], axis=1)
        left += h_T_W
        right += h_T_W

    return img[:, left:right].astype('uint8')


def create_gei_dataset(root_folder, save_path, T_H=64, T_W=64):
    data, labels, types, angles = [], [], [], []

    id_list = sorted(os.listdir(root_folder))

    for id_name in tqdm(id_list, desc="Processing IDs"):
        id_path = os.path.join(root_folder, id_name)
        if not os.path.isdir(id_path):
            continue

        skip_id = False

        for type_name in os.listdir(id_path):
            type_path = os.path.join(id_path, type_name)
            if not os.path.isdir(type_path):
                continue

            for angle_name in os.listdir(type_path):
                angle_path = os.path.join(type_path, angle_name)
                if not os.path.isdir(angle_path):
                    continue

                imgs = []
                for fname in sorted(os.listdir(angle_path)):
                    if fname.endswith(".png"):
                        fpath = os.path.join(angle_path, fname)
                        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                        img = pre_processing(img, T_H, T_W)
                        if img is not None:
                            imgs.append(img)

                if len(imgs) == 0:
                    skip_id = True
                    break 

                gei = np.mean(np.stack(imgs, axis=0), axis=0)
                gei = (gei / 255.0).astype('float32')  

                data.append(torch.from_numpy(gei))     
                labels.append(int(id_name))
                types.append(type_name)
                angles.append(angle_name)

            if skip_id:
                n_added = len([lbl for lbl in labels if lbl == int(id_name)])
                if n_added > 0:
                    data = data[:-n_added]
                    labels = labels[:-n_added]
                    types = types[:-n_added]
                    angles = angles[:-n_added]
                break

    dataset = {
        "data": torch.stack(data),              
        "labels": torch.tensor(labels),        
        "type": types,                          
        "angles": angles                       
    }

    torch.save(dataset, save_path)

# create_gei_dataset("GaitDatasetB-silh", "GEI.pt")
