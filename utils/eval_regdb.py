import os
"""RegDB evaluation helpers for single-gallery retrieval."""

import logging
import numpy as np
import torch

from torch.nn import functional as F

def pairwise_distance(x, y):
    """Return negative cosine similarity so lower values rank closer matches."""

    x = x.to("cuda")
    y = y.to("cuda")

    x = torch.nn.functional.normalize(x)
    y = torch.nn.functional.normalize(y)

    return (-x @ y.T).to("cpu")

def get_gallery_names(perm, cams, ids, trial_id, num_shots=1):
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

def eval_regdb(query_feats, query_ids, query_cam_ids, gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths):
    """Evaluate one RegDB direction; caller swaps query/gallery for the reverse direction."""

    gallery_feats = F.normalize(gallery_feats, dim=1)
    query_feats = F.normalize(query_feats, dim=1)

    dist_mat = pairwise_distance(query_feats, gallery_feats)

    sorted_indices = np.argsort(dist_mat, axis=1)

    mAP = get_mAP(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)
    cmc = get_cmc(sorted_indices, query_ids, query_cam_ids, gallery_ids, gallery_cam_ids)

    r1 = cmc[0]
    r5 = cmc[4]
    r10 = cmc[9]
    r20 = cmc[19]

    r1 = r1 * 100
    r5 = r5 * 100
    r10 = r10 * 100
    r20 = r20 * 100
    mAP = mAP * 100

    perf = "r1 precision = {:.2f} , r10 precision = {:.2f} , r20 precision = {:.2f}, mAP = {:.2f}"
    logging.info(perf.format(r1, r10, r20, mAP))

    return mAP, r1, r5, r10, r20
