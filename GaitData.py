import torch
from torch.utils.data import Dataset
from common import *

class GaitData(Dataset):
    def __init__(self, pt_path, labels, types=None, angles=None, map_label = MAP_TRAIN_LABEL):
        dataset = torch.load(pt_path)
        data = dataset["data"]
        all_labels = dataset["labels"]
        all_types = dataset["type"]
        all_angles = dataset["angles"]
        self.map_label = map_label

        self.samples = []
        for i in range(len(data)):
            t = all_types[i]
            a = all_angles[i]
            lbl = int(all_labels[i])

            if (types is None or t in types) and (angles is None or a in angles) and (labels is None or lbl in labels):
                self.samples.append({
                    "image": data[i],   
                    "label": lbl,
                    "type": t,
                    "angle": a
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return item["image"].unsqueeze(0), self.map_label[item["label"]]