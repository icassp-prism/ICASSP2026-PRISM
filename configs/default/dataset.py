"""Default dataset metadata and local data roots."""

from yacs.config import CfgNode

dataset_cfg = CfgNode()

dataset_cfg.sysu = CfgNode()
dataset_cfg.sysu.num_id = 395
dataset_cfg.sysu.data_root = "/home/zhutongqing/data/SYSU-MM01/SYSU-MM01"

dataset_cfg.regdb = CfgNode()
dataset_cfg.regdb.num_id = 206
dataset_cfg.regdb.data_root = "/home/zhutongqing/data/RegDB"

dataset_cfg.llcm = CfgNode()
dataset_cfg.llcm.num_id = 713
dataset_cfg.llcm.data_root = "/home/zhutongqing/data/LLCM"

dataset_cfg.cmgroup_crop = CfgNode()
dataset_cfg.cmgroup_crop.num_id = 233
dataset_cfg.cmgroup_crop.data_root = "/home/zhutongqing/data/CM-Group/CM-Group-crop"
