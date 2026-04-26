import os
import logging
import torch
import numpy as np
from torch.nn import functional as F
import pdb
import itertools
from scipy.optimize import linear_sum_assignment

def pairwise_distance(x, y):
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(x, y.t(), beta=1, alpha=-2)
    return dist

def ranking_base(query_feat, gallery_feat, query_ids, original_query_names, gallery_ids, original_gallery_names):
    """Baseline group distance by summing member features inside each group."""

    q_original_ids = np.unique(original_query_names)
    g_original_ids = np.unique(original_gallery_names)

    query_group_feats = []
    for q_id in q_original_ids:
        indices = np.where(original_query_names == q_id)[0]

        sum_feat = query_feat[indices].sum(dim=0)
        query_group_feats.append(sum_feat)

    gallery_group_feats = []
    for g_id in g_original_ids:
        indices = np.where(original_gallery_names == g_id)[0]

        sum_feat = gallery_feat[indices].sum(dim=0)
        gallery_group_feats.append(sum_feat)

    sum_query_feats_tensor = torch.stack(query_group_feats)
    sum_gallery_feats_tensor = torch.stack(gallery_group_feats)

    final_dist = pairwise_distance(sum_query_feats_tensor, sum_gallery_feats_tensor)

    return final_dist.numpy()

def ranking_cpd(query_feat, gallery_feat, query_ids, original_query_names, gallery_ids, original_gallery_names):
    """Legacy exhaustive CPD-style matching kept for comparison with MCBM."""

    dist = pairwise_distance(query_feat, gallery_feat)
    numpy_dist = dist.numpy()

    q_original_ids = np.unique(original_query_names)
    g_original_ids = np.unique(original_gallery_names)

    query_original_num = len(q_original_ids)
    gallery_original_num = len(g_original_ids)

    W = np.zeros((query_original_num, gallery_original_num)).astype(np.float16)

    q_id_to_indices = {}
    g_id_to_indices = {}

    for i, q_original_id in enumerate(q_original_ids):
        q_id_to_indices[q_original_id] = np.where(original_query_names == q_original_id)[0]

    for j, g_original_id in enumerate(g_original_ids):
        g_id_to_indices[g_original_id] = np.where(original_gallery_names == g_original_id)[0]

    for i in range(query_original_num):
        q_original_id = q_original_ids[i]
        q_indices = q_id_to_indices[q_original_id]
        q_id_num = len(q_indices)
        q_id_dist = numpy_dist[q_indices, :]

        for j in range(gallery_original_num):
            g_original_id = g_original_ids[j]
            g_indices = g_id_to_indices[g_original_id]
            g_id_num = len(g_indices)

            qg_dist = q_id_dist[:, g_indices]

            min_dist = float("inf")
            if q_id_num > g_id_num:
                for q_permutation in itertools.permutations(range(q_id_num), g_id_num):
                    dist_perm = [qg_dist[q_permutation[u]][u] for u in range(g_id_num)]
                    dist_summ = sum(dist_perm) + (q_id_num - g_id_num) * max(dist_perm)
                    min_dist = min(min_dist, dist_summ)
            else:
                for g_permutation in itertools.permutations(range(g_id_num), q_id_num):
                    dist_perm = [qg_dist[v][g_permutation[v]] for v in range(q_id_num)]
                    dist_summ = sum(dist_perm) + (g_id_num - q_id_num) * max(dist_perm)
                    min_dist = min(min_dist, dist_summ)

            W[i, j] = min_dist

    final_dist = W

    del dist
    del numpy_dist
    del W

    return final_dist

def minimum_cost_bipartite_matching(query_feat, gallery_feat, query_ids, original_query_names, gallery_ids, original_gallery_names):
    """Compute MCBM group distance with optimal matching and size penalty.

    Each query/gallery group is treated as a bipartite graph whose edge costs
    are member-level squared Euclidean distances. The Hungarian solver finds
    the minimum injective matching cost, and unmatched members are penalized by
    the largest matched-pair candidate cost to discourage size-mismatched groups.
    """

    cost_matrix = pairwise_distance(query_feat, gallery_feat).numpy()

    query_group_ids = np.unique(original_query_names)
    gallery_group_ids = np.unique(original_gallery_names)

    group_distance = np.zeros((len(query_group_ids), len(gallery_group_ids)), dtype=np.float32)

    query_group_indices = {group_id: np.where(original_query_names == group_id)[0] for group_id in query_group_ids}
    gallery_group_indices = {group_id: np.where(original_gallery_names == group_id)[0] for group_id in gallery_group_ids}

    for query_group_idx, query_group_id in enumerate(query_group_ids):
        query_indices = query_group_indices[query_group_id]
        query_group_cost = cost_matrix[query_indices, :]

        for gallery_group_idx, gallery_group_id in enumerate(gallery_group_ids):
            gallery_indices = gallery_group_indices[gallery_group_id]
            group_cost_matrix = query_group_cost[:, gallery_indices]

            # Eq. (9): solve the minimum-cost injective assignment between
            # group members. Eq. (10): add a size penalty for unmatched people.
            row_indices, col_indices = linear_sum_assignment(group_cost_matrix)
            matched_cost = group_cost_matrix[row_indices, col_indices].sum()
            size_penalty = abs(len(query_indices) - len(gallery_indices)) * np.max(group_cost_matrix)
            group_distance[query_group_idx, gallery_group_idx] = matched_cost + size_penalty

    return group_distance

def get_gallery_names(perm, cams, gp_ids, trial_id, num_shots=1):
    names = []
    for cam in cams:
        cam_perm = perm[cam - 1]
        for id in gp_ids:
            i = int(id.split("_")[0])
            instance_id = cam_perm[i - 1][trial_id][:num_shots]
            names.extend(["camera{}/{}/{}".format(cam, id, ins) for ins in instance_id])
    names = sorted(names)
    return names

def get_unique(array):
    _, idx = np.unique(array, return_index=True)
    return array[np.sort(idx)]

def get_cmc(sorted_indices, query_ids, gallery_ids):
    gallery_unique_count = get_unique(gallery_ids).shape[0]
    match_counter = np.zeros((gallery_unique_count,))

    result = gallery_ids[sorted_indices]

    valid_probe_sample_count = 0

    for probe_index in range(sorted_indices.shape[0]):
        result_i = result[probe_index, :]

        result_i_unique = get_unique(result_i)

        match_i = np.equal(result_i_unique, query_ids[probe_index])

        if np.sum(match_i) != 0:
            valid_probe_sample_count += 1
            match_counter += match_i

    rank = match_counter / valid_probe_sample_count
    cmc = np.cumsum(rank)
    return cmc

def get_mAP(sorted_indices, query_ids, gallery_ids):
    result = gallery_ids[sorted_indices]

    valid_probe_sample_count = 0
    avg_precision_sum = 0

    for probe_index in range(sorted_indices.shape[0]):
        result_i = result[probe_index, :]

        match_i = result_i == query_ids[probe_index]
        true_match_count = np.sum(match_i)

        if true_match_count != 0:
            valid_probe_sample_count += 1
            true_match_rank = np.where(match_i)[0]

            ap = np.mean(np.arange(1, true_match_count + 1) / (true_match_rank + 1))
            avg_precision_sum += ap

    mAP = avg_precision_sum / valid_probe_sample_count
    return mAP

def eval_cmgroup(query_feats, query_ids, query_img_paths, gallery_feats, gallery_ids, gallery_cam_ids, gallery_img_paths, perm, num_shots=1, num_trials=10, rank=0):
    """Evaluate CM-Group with optional baseline, CPD, or MCBM group ranking."""

    gallery_cams = [1, 2, 3]

    query_feats = F.normalize(query_feats, dim=1)
    original_query_names = np.array(["{}/{}/{}".format(path.split("/")[-3], path.split("/")[-2].split("_")[0], path.split("/")[-1].split(".")[0]) for path in query_img_paths])
    q_original_gids = np.array([int(name.split("/")[1]) for name in np.unique(original_query_names)])

    gallery_indices = np.isin(gallery_cam_ids, gallery_cams)

    gallery_feats = gallery_feats[gallery_indices]
    gallery_feats = F.normalize(gallery_feats, dim=1)

    gallery_cam_ids = gallery_cam_ids[gallery_indices]
    gallery_ids = gallery_ids[gallery_indices]

    gallery_img_paths = gallery_img_paths[gallery_indices]
    gallery_names = np.array(["/".join(os.path.splitext(path)[0].split("/")[-3:]) for path in gallery_img_paths])
    original_gallery_names = np.array(["{}/{}/{}".format(path.split("/")[-3], path.split("/")[-2].split("_")[0], path.split("/")[-1].split(".")[0]) for path in gallery_img_paths])

    gallery_gp_ids = [path.split("/")[-2] for path in gallery_img_paths]
    gallery_gp_id_set = np.unique(gallery_gp_ids)

    mAP, r1, r5, r10, r20 = 0, 0, 0, 0, 0
    for t in range(num_trials):
        names = get_gallery_names(perm, gallery_cams, gallery_gp_id_set, t, num_shots)
        flag = np.isin(gallery_names, names)

        g_feat = gallery_feats[flag]
        g_ids = gallery_ids[flag]
        g_original_names = original_gallery_names[flag]
        g_original_gids = np.array([int(name.split("/")[1]) for name in np.unique(g_original_names)])

        # rank selects the group distance implementation: 1=sum baseline, 2=CPD, else=MCBM.
        if rank == 1:
            dist_mat = ranking_cpd(query_feats, g_feat, query_ids, original_query_names, g_ids, g_original_names)
        elif rank == 2:
            dist_mat = ranking_base(query_feats, g_feat, query_ids, original_query_names, g_ids, g_original_names)
        else:
            dist_mat = minimum_cost_bipartite_matching(query_feats, g_feat, query_ids, original_query_names, g_ids, g_original_names)

        sorted_indices = np.argsort(dist_mat, axis=1)

        mAP += get_mAP(sorted_indices, q_original_gids, g_original_gids)
        cmc = get_cmc(sorted_indices, q_original_gids, g_original_gids)

        r1 += cmc[0]
        r5 += cmc[4]
        r10 += cmc[9]
        r20 += cmc[19]

    r1 = r1 / num_trials * 100
    r5 = r5 / num_trials * 100
    r10 = r10 / num_trials * 100
    r20 = r20 / num_trials * 100
    mAP = mAP / num_trials * 100

    perf = "num-shot:{} r1 precision = {:.2f} , r10 precision = {:.2f} , r20 precision = {:.2f}, mAP = {:.2f}"
    logging.info(perf.format(num_shots, r1, r10, r20, mAP))

    return mAP, r1, r5, r10, r20
