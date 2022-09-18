
from get_logger import logger
import torch
import os, pdb, glob, random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from .DataAug import DA_Rotation
rootpath = "/home/bebin.huang/Code/FoG_prediction/FoG_datasets2"

class FoG_dataset(Dataset):
    def __init__(self, samples: list, rootpth = None, level="episode", mode="train", 
                aug=False) -> None:
        super().__init__()
        assert mode in ["train", "val"]
        self.rootpath = rootpath
        if rootpth is not None:
            self.rootpath = rootpth
        self.samples = samples
        self.level = level
        self.mode = mode
        self.aug = aug
        # pdb.set_trace()
        if level == "episode":
            sam_all = []
            for sam in samples:
                sam_all.extend([f.split("/")[-1] for f in glob.glob(os.path.join(self.rootpath, sam+"*"))])
            self.samples = sam_all
        self.cls_weight = self.log()

    def log(self):
        total = len(self.samples)
        NW, preFoG, FoG = 0, 0, 0
        for sam in self.samples:
            if sam[-5] == "0":
                NW += 1
            elif sam[-5] == "1":
                preFoG += 1
            elif sam[-5] == "2":
                FoG += 1
            else:
                raise ValueError("Label error: must be 0, 1 or 2.")
        ratio = [preFoG*FoG, NW*FoG, NW*preFoG] # NW:preFoG:FoG
        ratio = [r/sum(ratio) for r in ratio]
        logger.info("Mode = {}, aug = {}, total = {}, NW = [{}, {:.2f}], preFoG = [{}, {:.2f}], FoG = [{}, {:.2f}]".format(
            self.mode, self.aug, total, NW, ratio[0], preFoG, ratio[1], FoG, ratio[2]
        ))
        return torch.Tensor(ratio)

    def __getitem__(self, index):
        data = np.loadtxt(os.path.join(self.rootpath, self.samples[index]))
        data = torch.FloatTensor(data)
        label = self.samples[index]
        if self.aug:
            data_aug = DA_Rotation(data)
            return [data, data_aug], [label, label]
        
        return data, label

    def __len__(self):
        return len(self.samples)

def collate_fn(batch: list):
    data, labels, info = [], [], []
    for d, lab in batch:
        if isinstance(d, list):
            data.extend(d)
            info.extend(lab)
            labels.extend([int(l.split("_")[-1].split(".")[0]) for l in lab])
        else:
            data.append(d)
            info.append(lab)
            labels.append(int(lab.split("_")[-1].split(".")[0]))
    data = torch.stack(data, dim=0)
    labels = torch.LongTensor(labels)
    return data, {"info": info, "labels": labels}

def split_n_fold(n=10, rootpth = None, level="episode", seed=10086): # level = episode or sample
    random.seed(seed)
    samples_all = os.listdir(rootpath if rootpth is None else rootpth)
    episodes_all = list(set([sam[:11] for sam in samples_all]))
    logger.info("total samples: {}, total episodes: {}".format(len(samples_all), len(episodes_all)))

    if level == "episode":
        epi_idx = list(range(len(episodes_all)))
        random.shuffle(epi_idx)
        eachfold = len(episodes_all) // n
        parts, foldidx = [], eachfold
        while foldidx < len(episodes_all):
            parts.append([episodes_all[epi_idx[i]] for i in range(foldidx-eachfold, foldidx)])
            foldidx += eachfold
        for i in range(foldidx-eachfold, len(episodes_all)):
            parts[i-(foldidx-eachfold)].append(episodes_all[epi_idx[i]])
        return parts
    elif level == "sample":
        sam_idx = list(range(len(samples_all)))
        random.shuffle(sam_idx)
        eachfold = len(samples_all) // n
        parts, foldidx = [], eachfold
        while foldidx < len(samples_all):
            parts.append([samples_all[sam_idx[i]] for i in range(foldidx-eachfold, foldidx)])
            foldidx += eachfold
        for i in range(foldidx-eachfold, len(samples_all)):
            parts[i-(foldidx-eachfold)].append(samples_all[sam_idx[i]])
        return parts
    else:
        raise ValueError("level error!")

def build_dataset(samples, rootpth = None, level="episode", mode="train", aug=False):
    return FoG_dataset(samples, rootpth = rootpth, level=level, mode=mode, aug=aug)

if __name__ == "__main__":
    pdb.set_trace()
    level = "episode"
    parts = split_n_fold(level=level)
    dataset = FoG_dataset(parts[0], level=level)
    dataloader = DataLoader(dataset, 8, shuffle=True, collate_fn=collate_fn)

    for i, (data, labels) in enumerate(dataloader):
        print(data.shape, labels.shape)
    # for i in range(3):
    #     data, label = dataset[i]
    # f = 1