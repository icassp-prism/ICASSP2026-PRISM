"""Default training strategy values merged with dataset and YAML overrides."""

from yacs.config import CfgNode

strategy_cfg = CfgNode()

strategy_cfg.prefix = "baseline"

strategy_cfg.sample_method = "random"
strategy_cfg.batch_size = 128
strategy_cfg.p_size = 16
strategy_cfg.k_size = 8

strategy_cfg.optimizer = "sgd"
strategy_cfg.lr = 0.1
strategy_cfg.wd = 5e-4
strategy_cfg.lr_step = [40]

strategy_cfg.num_epoch = 60

strategy_cfg.dataset = "sysu"
strategy_cfg.image_size = (288, 144)

strategy_cfg.random_flip = True
strategy_cfg.random_crop = True
strategy_cfg.random_erase = True
strategy_cfg.color_jitter = False
strategy_cfg.padding = 10

strategy_cfg.modality_attention = 0
strategy_cfg.mutual_learning = False

strategy_cfg.eval_interval = -1
strategy_cfg.start_eval = 60

strategy_cfg.resume = ''

strategy_cfg.log_name = "log.txt"
