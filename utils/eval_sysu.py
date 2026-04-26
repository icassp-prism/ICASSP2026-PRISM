"""SYSU-MM01 evaluation helpers with the official multi-shot protocol."""

import os
import logging
import torch
import numpy as np

from torch.nn import functional as F

def pairwise_distance(x, y):
    """Return negative cosine similarity so lower values rank closer matches."""

    x = x.to("cuda")
    y = y.to("cuda")

    x = torch.nn.functional.normalize(x)
    y = torch.nn.functional.normalize(y)

    return (-x @ y.T).to("cpu")

def get_gallery_names(perm, cams, ids, trial_id, num_shots=1):
    """Select the official SYSU trial gallery names from rand_perm_cam.mat."""

    names = []
    for cam in cams:
        cam_perm = perm[cam - 1][0].squeeze()
        for i in ids:
            instance_id = cam_perm[i - 1][trial_id][:num_shots]
            names.extend(["cam{}/{:0>4d}/{:0>4d}".format(cam, i, ins) for ins in instance_id.tolist()])

    return names

def get_unique(array):
    _, idx = np.unique(array, return_index=True)
    return array[np.sort(idx)]

def get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    """Compute CMC after removing invalid same-camera matches for each query."""

    max_rank = gallery_ids.shape[0]
    match_counter = np.zeros((max_rank,))

    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0

    for probe_index in range(sorted_indices.shape[0]):
        result_i = result[probe_index, :]
        result_i[np.equal(cam_locations_result[probe_index], query_cam_ids[probe_index])] = -1

        result_i = np.array([i for i in result_i if i != -1])

        result_i_unique = get_unique(result_i)

        match_i = np.equal(result_i_unique, query_ids[probe_index])

        if np.sum(match_i) != 0:
            valid_probe_sample_count += 1

            first_match_rank = np.flatnonzero(match_i)[0]
            match_counter[first_match_rank] += 1

    rank = match_counter / valid_probe_sample_count
    cmc = np.cumsum(rank)
    return cmc

def get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids):
    """Compute AP per valid query using the dataset-specific filtered gallery list."""

    result = gallery_ids[sorted_indices]
    cam_locations_result = gallery_cam_ids[sorted_indices]

    valid_probe_sample_count = 0
    avg_precision_sum = 0

    for probe_index in range(sorted_indices.shape[0]):
        result_i = result[probe_index, :]
        result_i[cam_locations_result[probe_index, :] == query_cam_ids[probe_index]] = -1

        result_i = np.array([i for i in result_i if i != -1])

        match_i = result_i == query_ids[probe_index]
        true_match_count = np.sum(match_i)

        if true_match_count != 0:
            valid_probe_sample_count += 1
            true_match_rank = np.where(match_i)[0]

            ap = np.mean(np.arange(1, true_match_count + 1) / (true_match_rank + 1))
            avg_precision_sum += ap

    mAP = avg_precision_sum / valid_probe_sample_count
    return mAP

def eval_sysu(query_feats, query_ids, query_cam_ids, gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths, perm, mode="all", num_shots=1, num_trials=10):
    """Average SYSU metrics over the official gallery sampling trials."""

    assert mode in ["indoor", "all"]

    gallery_cams = [1, 2] if mode == "indoor" else [1, 2, 4, 5]

    # SYSU treats cam2 and cam3 as the same location for same-camera filtering.
    query_cam_ids[np.equal(query_cam_ids, 3)] = 2
    query_feats = F.normalize(query_feats, dim=1)

    gallery_indices = np.in1d(gallery_cam_ids, gallery_cams)

    gallery_feats = gallery_feats[gallery_indices]
    gallery_feats = F.normalize(gallery_feats, dim=1)
    gallery_cam_ids = gallery_cam_ids[gallery_indices]
    gallery_ids = gallery_ids[gallery_indices]
    gallery_img_paths = gallery_img_paths[gallery_indices]
    gallery_names = np.array(["/".join(os.path.splitext(path)[0].split("/")[-3:]) for path in gallery_img_paths])

    gallery_id_set = np.unique(gallery_ids)

    mAP, r1, r5, r10, r20 = 0, 0, 0, 0, 0
    for t in range(num_trials):
        names = get_gallery_names(perm, gallery_cams, gallery_id_set, t, num_shots)
        flag = np.in1d(gallery_names, names)

        g_feat = gallery_feats[flag]
        g_ids = gallery_ids[flag]
        g_cam_ids = gallery_cam_ids[flag]

        dist_mat = pairwise_distance(query_feats, g_feat)

        sorted_indices = np.argsort(dist_mat, axis=1)

        mAP += get_mAP(sorted_indices, query_ids, query_cam_ids, g_ids, g_cam_ids)
        cmc = get_cmc(sorted_indices, query_ids, query_cam_ids, g_ids, g_cam_ids)

        r1 += cmc[0]
        r5 += cmc[4]
        r10 += cmc[9]
        r20 += cmc[19]

    r1 = r1 / num_trials * 100
    r5 = r5 / num_trials * 100
    r10 = r10 / num_trials * 100
    r20 = r20 / num_trials * 100
    mAP = mAP / num_trials * 100

    perf = "{} num-shot:{} r1 precision = {:.2f} , r10 precision = {:.2f} , r20 precision = {:.2f}, mAP = {:.2f}"
    logging.info(perf.format(mode, num_shots, r1, r10, r20, mAP))

    return mAP, r1, r5, r10, r20
