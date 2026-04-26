"""Dataset wrappers normalize IDs, cameras, modalities, and image paths."""

import os
import re
import os.path as osp
from glob import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class SYSUDataset(Dataset):
    """SYSU-MM01 split parser with visible/infrared modality labels."""
    def __init__(self, root, mode="train", transform=None):
        assert os.path.isdir(root)
        assert mode in ["train", "gallery", "query"]

        if mode == "train":
            train_ids = open(os.path.join(root, "exp", "train_id.txt")).readline()
            val_ids = open(os.path.join(root, "exp", "val_id.txt")).readline()

            train_ids = train_ids.strip("\n").split(",")
            val_ids = val_ids.strip("\n").split(",")
            selected_ids = train_ids + val_ids
        else:
            test_ids = open(os.path.join(root, "exp", "test_id.txt")).readline()
            selected_ids = test_ids.strip("\n").split(",")

        selected_ids = [int(i) for i in selected_ids]
        num_ids = len(selected_ids)

        img_paths = glob(os.path.join(root, "**/*.jpg"), recursive=True)
        img_paths = [path for path in img_paths if int(path.split("/")[-2]) in selected_ids]

        if mode == "gallery":
            img_paths = [path for path in img_paths if int(path.split("/")[-3][-1]) in (1, 2, 4, 5)]
        elif mode == "query":
            img_paths = [path for path in img_paths if int(path.split("/")[-3][-1]) in (3, 6)]

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        self.cam_ids = [int(path.split("/")[-3][-1]) for path in img_paths]
        self.modal_ids = [int(cam in (3, 6)) for cam in self.cam_ids]
        self.num_ids = num_ids
        self.transform = transform

        if mode == "train":
            # Relabel training identities to contiguous IDs for classification loss.
            id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [id_map[int(path.split("/")[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split("/")[-2]) for path in img_paths]

        self.index2img = {}

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = self.__getImage(item)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        modal = torch.tensor(self.modal_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, modal, path, item

    def __getImage(self, index):
        if index not in self.index2img:
            path = self.img_paths[index]
            img = Image.open(path)
            self.index2img[index] = img
        else:
            img = self.index2img[index]

        return img

class RegDBDataset(Dataset):
    """RegDB split parser; visible and thermal folders define modality IDs."""
    def __init__(self, root, mode="train", transform=None, split_num="10"):
        assert os.path.isdir(root)
        assert mode in ["train", "gallery", "query"]

        def loadIdx(index):
            Lines = index.readlines()
            idx = []
            for line in Lines:
                tmp = line.strip("\n")
                tmp = tmp.split(" ")
                idx.append(tmp)
            return idx

        num = split_num
        if mode == "train":
            index_RGB = loadIdx(open(root + "/idx/train_visible_" + num + ".txt", "r"))
            index_IR = loadIdx(open(root + "/idx/train_thermal_" + num + ".txt", "r"))
        else:
            index_RGB = loadIdx(open(root + "/idx/test_visible_" + num + ".txt", "r"))
            index_IR = loadIdx(open(root + "/idx/test_thermal_" + num + ".txt", "r"))

        if mode == "gallery":
            img_paths = [root + "/" + path for path, _ in index_RGB]
        elif mode == "query":
            img_paths = [root + "/" + path for path, _ in index_IR]
        else:
            img_paths = [root + "/" + path for path, _ in index_RGB] + [root + "/" + path for path, _ in index_IR]

        selected_ids = [int(path.split("/")[-2]) for path in img_paths]
        selected_ids = list(set(selected_ids))
        num_ids = len(selected_ids)

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        # RegDB has no SYSU-style camera IDs; map folders to stable pseudo-camera IDs.
        self.cam_ids = [int(path.split("/")[-3] == "Thermal") + 2 for path in img_paths]

        self.modal_ids = [int(path.split("/")[-3] == "Thermal") for path in img_paths]
        self.num_ids = num_ids
        self.transform = transform

        if mode == "train":
            id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [id_map[int(path.split("/")[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split("/")[-2]) for path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        modal = torch.tensor(self.modal_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, modal, path, item

class LLCMDataset(Dataset):
    """LLCM split parser with RGB/NIR camera and modality conventions."""
    def __init__(self, root, mode="train", transform=None):
        assert os.path.isdir(root)
        assert mode in ["train", "gallery", "query"]

        def loadIdx(index):
            Lines = index.readlines()
            idx = []
            for line in Lines:
                tmp = line.strip("\n")
                tmp = tmp.split(" ")
                idx.append(tmp)
            return idx

        if mode == "train":
            index_RGB = loadIdx(open(root + "/idx/train_vis.txt", "r"))
            index_IR = loadIdx(open(root + "/idx/train_nir.txt", "r"))
        else:
            index_RGB = loadIdx(open(root + "/idx/test_vis.txt", "r"))
            index_IR = loadIdx(open(root + "/idx/test_nir.txt", "r"))

        if mode == "gallery":
            img_paths = [root + "/" + path for path, _ in index_RGB]
        elif mode == "query":
            img_paths = [root + "/" + path for path, _ in index_IR]
        else:
            img_paths = [root + "/" + path for path, _ in index_RGB] + [root + "/" + path for path, _ in index_IR]

        selected_ids = [int(path.split("/")[-2]) for path in img_paths]
        selected_ids = list(set(selected_ids))
        num_ids = len(selected_ids)

        img_paths = sorted(img_paths)
        self.img_paths = img_paths
        # LLCM camera IDs are encoded in filenames, while modality comes from folder names.
        self.cam_ids = [int(re.search(r"_c(\d+)", osp.basename(path)).group(1)) for path in img_paths]

        self.modal_ids = [int(path.split("/")[-3] == "nir") for path in img_paths]

        self.num_ids = num_ids
        self.transform = transform

        if mode == "train":
            id_map = dict(zip(selected_ids, range(num_ids)))
            self.ids = [id_map[int(path.split("/")[-2])] for path in img_paths]
        else:
            self.ids = [int(path.split("/")[-2]) for path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        modal = torch.tensor(self.modal_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, modal, path, item

class CMGroupDataset(Dataset):
    """CM-Group parser that keeps group identity names for group matching."""
    def __init__(self, root, mode="train", transform=None):
        assert os.path.isdir(root)
        assert mode in ["train", "gallery", "query"]

        if mode == "train":
            train_gids = np.load(os.path.join(root, "npy", "train_id.npy")).tolist()
            selected_gids = train_gids
        else:
            test_gids = np.load(os.path.join(root, "npy", "test_id.npy")).tolist()
            selected_gids = test_gids

        img_paths = glob(os.path.join(root, "**/*.jpeg"), recursive=True)
        img_paths = [path for path in img_paths if int(path.split("/")[-2].split("_")[0]) in selected_gids]

        if mode == "gallery":
            img_paths = [path for path in img_paths if int(path.split("/")[-3][-1]) in (1, 2, 3)]
        elif mode == "query":
            img_paths = [path for path in img_paths if int(path.split("/")[-3][-1]) in (4, 5, 6)]

        img_paths = sorted(img_paths)
        # Group IDs are directory names; the suffix is used as evaluation identity.
        gp_ids = [path.split("/")[-2] for path in img_paths]
        gp_ids = np.unique(gp_ids)
        num_ids = len(gp_ids)

        self.img_paths = img_paths
        self.cam_ids = [int(path.split("/")[-3][-1]) for path in img_paths]
        self.modal_ids = [int(cam in (4, 5, 6)) for cam in self.cam_ids]
        self.num_ids = num_ids
        self.transform = transform

        if mode == "train":
            id_map = dict(zip(gp_ids, range(num_ids)))
            self.ids = [id_map[path.split("/")[-2]] for path in img_paths]
        else:
            self.ids = [int(path.split("/")[-2].split("_")[-1]) for path in img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        path = self.img_paths[item]

        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(self.ids[item], dtype=torch.long)
        cam = torch.tensor(self.cam_ids[item], dtype=torch.long)
        modal = torch.tensor(self.modal_ids[item], dtype=torch.long)
        item = torch.tensor(item, dtype=torch.long)

        return img, label, cam, modal, path, item
