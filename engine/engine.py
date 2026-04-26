"""Ignite train/eval step functions shared by all dataset runners."""

import torch
import numpy as np

from ignite.engine import Engine
from ignite.engine import Events
from torch.autograd import no_grad

def create_train_engine(model, optimizer, non_blocking=False):
    """Create an Ignite trainer that expects model outputs as loss and metric dict."""

    device = torch.device("cuda", torch.cuda.current_device())
    scaler = torch.amp.GradScaler("cuda")

    def _process_func(engine, batch):
        model.train()

        data, labels, _, modal_ids, _, _ = batch
        epoch = engine.state.epoch

        data = data.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        modal_ids = modal_ids.to(device, non_blocking=non_blocking)

        optimizer.zero_grad()
        with torch.autocast("cuda", torch.float16):
            loss, metric = model(data, labels, modal_ids=modal_ids, epoch=epoch)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        return metric

    return Engine(_process_func)

def create_eval_engine(model, non_blocking=False):
    """Create an Ignite evaluator that accumulates features, IDs, cameras, and paths."""

    device = torch.device("cuda", torch.cuda.current_device())

    def _process_func(engine, batch):
        model.eval()

        data, labels, cam_ids, modal_ids, img_paths, _ = batch

        data = data.to(device, non_blocking=non_blocking)

        with no_grad():
            # Evaluation mode returns BN-neck features instead of training losses.
            feat = model(data, modal_ids=modal_ids.to(device, non_blocking=non_blocking))

        return feat.data.float().cpu(), labels, cam_ids, np.array(img_paths)

    engine = Engine(_process_func)

    @engine.on(Events.EPOCH_STARTED)
    def clear_data(engine):
        if not hasattr(engine.state, "feat_list"):
            setattr(engine.state, "feat_list", [])
        else:
            engine.state.feat_list.clear()

        if not hasattr(engine.state, "id_list"):
            setattr(engine.state, "id_list", [])
        else:
            engine.state.id_list.clear()

        if not hasattr(engine.state, "cam_list"):
            setattr(engine.state, "cam_list", [])
        else:
            engine.state.cam_list.clear()

        if not hasattr(engine.state, "img_path_list"):
            setattr(engine.state, "img_path_list", [])
        else:
            engine.state.img_path_list.clear()

    @engine.on(Events.ITERATION_COMPLETED)
    def store_data(engine):
        engine.state.feat_list.append(engine.state.output[0])
        engine.state.id_list.append(engine.state.output[1])
        engine.state.cam_list.append(engine.state.output[2])
        engine.state.img_path_list.append(engine.state.output[3])

    return engine
