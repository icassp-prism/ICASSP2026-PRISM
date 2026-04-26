"""Command-line entrypoint for model training/evaluation."""

import logging
import os
import pprint

import torch

from torch import optim

from data import get_test_loader
from data import get_train_loader
from engine import get_trainer
from models.baseline import Baseline

def train(cfg):
    """Build loaders, model, optimizer, and Ignite engine from the merged config."""

    log_dir = os.path.join("logs/", cfg.dataset, cfg.prefix)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s %(message)s",
        filename=log_dir + "/" + cfg.log_name,
        filemode="w",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    logger.info(pprint.pformat(cfg))

    train_loader = get_train_loader(
        dataset=cfg.dataset,
        root=cfg.data_root,
        sample_method=cfg.sample_method,
        batch_size=cfg.batch_size,
        p_size=cfg.p_size,
        k_size=cfg.k_size,
        random_flip=cfg.random_flip,
        random_crop=cfg.random_crop,
        random_erase=cfg.random_erase,
        color_jitter=cfg.color_jitter,
        padding=cfg.padding,
        image_size=cfg.image_size,
        num_workers=8,
    )

    gallery_loader, query_loader = None, None
    if cfg.eval_interval > 0:
        gallery_loader, query_loader = get_test_loader(
            dataset=cfg.dataset,
            root=cfg.data_root,
            batch_size=64,
            image_size=cfg.image_size,
            num_workers=4,
        )

    model = Baseline(
        dataset=cfg.dataset,
        num_classes=cfg.num_id,
        modality_attention=cfg.modality_attention,
        mutual_learning=cfg.mutual_learning,
    )

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {"Total": total_num, "Trainable": trainable_num}

    print(get_parameter_number(model))

    model.cuda()

    if cfg.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)

    def step_lr_with_warmup(epoch):
        if epoch < 10:
            return (epoch + 1) / 10
        else:
            if epoch < cfg.lr_step[0]:
                return 1
            elif epoch < cfg.lr_step[1]:
                return 0.1
            else:
                return 0.01

    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=step_lr_with_warmup)

    if cfg.resume:
        checkpoint = torch.load(cfg.resume)
        model.load_state_dict(checkpoint)

    checkpoint_dir = os.path.join("checkpoints", cfg.dataset, cfg.prefix)
    engine = get_trainer(
        dataset=cfg.dataset,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        logger=logger,
        non_blocking=True,
        save_dir=checkpoint_dir,
        prefix=cfg.prefix,
        eval_interval=cfg.eval_interval,
        start_eval=cfg.start_eval,
        gallery_loader=gallery_loader,
        query_loader=query_loader,
    )

    engine.run(train_loader, max_epochs=cfg.num_epoch)

if __name__ == "__main__":
    import argparse
    import random
    import numpy as np
    from configs.default import strategy_cfg
    from configs.default import dataset_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/SYSU.yml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_name", type=str, default="log.txt")
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.RandomState(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = True

    cfg = strategy_cfg
    # YAML config overrides defaults; CLI log_name is merged after the file.
    cfg.merge_from_file(args.cfg)

    dataset_cfg = dataset_cfg.get(cfg.dataset)

    for k, v in dataset_cfg.items():
        cfg[k] = v

    if cfg.sample_method == "identity_uniform":
        cfg.batch_size = cfg.p_size * cfg.k_size

    cfg.log_name = args.log_name
    cfg.prefix = f"{cfg.prefix}_{cfg.log_name}"
    cfg.freeze()

    train(cfg)
