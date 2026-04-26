import pdb
import torch
import torch.nn as nn
from torch.nn import init

from models.backbone import Backbone

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
        init.zeros_(m.bias.data)
    elif classname.find("BatchNorm1d") != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

class CenterSeparationLoss(nn.Module):
    """Mini-batch implementation of Lcs from the paper.

    The loss pulls each feature toward the center of its current-batch identity
    and pushes different identity centers apart. It intentionally uses batch
    centers instead of a persistent center table, so it depends on the identity
    sampler providing multiple samples per identity.
    """

    def __init__(self, k_size, margin1=0, margin2=0.7):
        super(CenterSeparationLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2
        self.k_size = k_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin2)

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Build a per-sample class center from the current mini-batch.
        centers = []
        for i in range(n):
            centers.append(inputs[targets == targets[i]].mean(0))
        centers = torch.stack(centers)

        dist_pc = (inputs - centers) ** 2
        dist_pc = dist_pc.sum(1)
        dist_pc = dist_pc.sqrt()
        dist_pc = (dist_pc - self.margin1).clamp(min=0.0)

        # Penalize class centers that are closer than margin2.
        dist = torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(centers, centers.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_an = []
        for i in range(0, n, self.k_size):
            dist_an.append((self.margin2 - dist[i][mask[i] == 0]).clamp(min=0.0).mean())
        dist_an = torch.stack(dist_an)

        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)

        loss = dist_pc.mean() + dist_an.mean()

        return loss

class ModalitySpecificBNNeck(nn.Module):
    """Use separate BN statistics for visible and infrared features."""

    def __init__(self, dim) -> None:
        super().__init__()

        self.bn_neck_v = nn.BatchNorm1d(dim)
        self.bn_neck_i = nn.BatchNorm1d(dim)
        nn.init.constant_(self.bn_neck_i.bias, 0)
        nn.init.constant_(self.bn_neck_v.bias, 0)
        self.bn_neck_v.bias.requires_grad_(False)
        self.bn_neck_i.bias.requires_grad_(False)

    def forward(self, features, modality_ids):
        infrared_mask = modality_ids == 1
        visible_mask = modality_ids == 0

        features[infrared_mask] = self.bn_neck_i(features[infrared_mask])
        features[visible_mask] = self.bn_neck_v(features[visible_mask])

        return features

class MASA(nn.Module):
    """Modality-Aware Semantic Alignment from PRISM.

    MASA learns a shared semantic prototype base Pb plus modality-specific
    offsets Om. For each sample, Pm = Pb + Om generates prototypes adapted to
    visible or infrared inputs. These prototypes attend over pixel-level
    backbone features to produce semantic part features pm, which are later
    concatenated with the global average pooled feature Fg_m.
    """

    def __init__(self, dim=2048, part_num=6, h=18, w=9):
        super().__init__()
        self.part_num = part_num
        self.h = h
        self.w = w

        # Pb: shared semantic anchors. They are not tied to fixed body positions;
        # permutation-invariant training below forces each anchor to learn semantics.
        self.shared_prototypes_base = nn.Parameter(nn.init.kaiming_normal_(torch.empty(part_num, dim)))

        # Om: modality-specific offsets compensate for visible/infrared appearance gaps
        # while keeping both modalities anchored to the same semantic prototype base.
        self.visible_modality_offset = nn.Parameter(nn.init.kaiming_normal_(torch.empty(part_num, dim)))
        self.infrared_modality_offset = nn.Parameter(nn.init.kaiming_normal_(torch.empty(part_num, dim)))

        self.position_embedding = nn.Parameter(nn.init.kaiming_normal_(torch.empty(h * w, dim)))

        self.active = nn.Sigmoid()

    def forward(self, feature_map, modality_ids):
        B, C, H, W = feature_map.shape
        pixel_features = feature_map.view(B, C, -1).permute(0, 2, 1)
        position_aware_features = pixel_features + self.position_embedding

        visible_mask = (modality_ids == 0).unsqueeze(1).unsqueeze(2).float()
        infrared_mask = (modality_ids == 1).unsqueeze(1).unsqueeze(2).float()

        # Pm = Pb + Om is generated per sample. Broadcasting keeps a separate
        # prototype set for each image in a mixed visible/infrared mini-batch.
        modality_aware_prototypes = self.shared_prototypes_base.unsqueeze(0) + visible_mask * self.visible_modality_offset.unsqueeze(0) + infrared_mask * self.infrared_modality_offset.unsqueeze(0)

        if self.training:
            # Randomly permute prototypes to remove dependency on prototype index/order.
            # After aggregation, the inverse permutation restores a stable feature layout.
            rand_idx = torch.randperm(self.part_num).to(modality_aware_prototypes.device)

            permuted_prototypes = modality_aware_prototypes[:, rand_idx, :]
        else:
            permuted_prototypes = modality_aware_prototypes

        # S_m = sigmoid(P_m @ X_m^T): each prototype produces one attention map
        # over all H*W pixel features. Position embeddings preserve coarse layout cues.
        attn = torch.bmm(permuted_prototypes, position_aware_features.transpose(1, 2))
        attn = self.active(attn)

        # p_m = S_m @ X_m / n: attention-weighted pooling converts pixel features
        # into part-level semantic features without hand-crafted body partitions.
        permuted_part_features = torch.bmm(attn, pixel_features) / H / W

        if self.training:
            # Restore the original prototype order so concatenated features stay stable.
            inv_rand_idx = torch.argsort(rand_idx)

            part_features = torch.stack([shuffled[inv_rand_idx] for shuffled in permuted_part_features])
            attention_scores = torch.stack([a[inv_rand_idx] for a in attn])
        else:
            part_features = permuted_part_features
            attention_scores = attn

        semantic_part_features = part_features.contiguous().view(B, -1)

        return semantic_part_features, attention_scores

class Baseline(nn.Module):
    """Full PRISM-style model: backbone, MASA features, BN neck, and losses."""
    def __init__(self, num_classes=None, dataset=None, modality_attention=0, mutual_learning=False):
        super().__init__()

        self.dataset = dataset
        if self.dataset == "cmgroup_crop":
            self.h = 24
            self.w = 8
        else:
            self.h = 18
            self.w = 9

        self.mutual_learning = mutual_learning

        self.backbone = Backbone(num_classes=num_classes, modality_attention=modality_attention)

        D = 2048
        self.base_dim = D
        self.dim = D
        self.part_num = 7

        self.k_size = 8
        self.margin1 = 0.01
        self.margin2 = 0.7

        self.masa = MASA(dim=self.base_dim, part_num=self.part_num, h=self.h, w=self.w)
        self.bn_neck = ModalitySpecificBNNeck(self.base_dim + self.dim * self.part_num)

        self.visible_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.infrared_classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.visible_classifier_ = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.visible_classifier_.weight.requires_grad_(False)
        self.visible_classifier_.weight.data = self.visible_classifier.weight.data
        self.infrared_classifier_ = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)
        self.infrared_classifier_.weight.requires_grad_(False)
        self.infrared_classifier_.weight.data = self.infrared_classifier.weight.data
        self.update_rate = 0.2
        self.update_rate_ = self.update_rate

        self.classifier = nn.Linear(self.base_dim + self.dim * self.part_num, num_classes, bias=False)

        self.center_separation_loss = CenterSeparationLoss(k_size=self.k_size, margin1=self.margin1, margin2=self.margin2)
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, inputs, labels=None, **kwargs):
        modality_ids = kwargs.get("modal_ids")

        feature_map = self.backbone(inputs, modality_ids)
        batch_size, _, _, _ = feature_map.shape

        semantic_part_features, semantic_attention = self.masa(feature_map, modality_ids)
        global_features = feature_map.mean(dim=(2, 3))
        # Final representation follows Fm = [pm, Fg_m] from the MASA section.
        feature_representation = torch.cat([semantic_part_features, global_features], dim=1)

        if self.training:
            metric = {}
            loss = 0.0

            visible_mask = modality_ids == 0
            infrared_mask = modality_ids == 1

            # Ldp maximizes pairwise distances between attention maps. Without it,
            # several prototypes can collapse onto the same discriminative region.
            attention_maps = semantic_attention.view(batch_size, self.part_num, self.h * self.w)
            loss_part_diversity = 0.0
            for i in range(self.part_num):
                for j in range(i + 1, self.part_num):
                    loss_part_diversity += ((((attention_maps[:, i] - attention_maps[:, j]) ** 2).sum(dim=1) / (self.h * self.w)) ** 0.5).sum()
            loss_part_diversity = -loss_part_diversity / (batch_size * self.part_num * (self.part_num - 1) / 2)
            loss_part_diversity *= 0.5
            metric.update({"dp": loss_part_diversity.data})
            loss += loss_part_diversity * 0.5

            loss_center_separation = self.center_separation_loss(feature_representation.float(), labels)
            metric.update({"cs": loss_center_separation.data})
            loss += loss_center_separation * 1.2

            feature_representation = self.bn_neck(feature_representation, modality_ids)
            logits = self.classifier(feature_representation)
            loss_cross_entropy = self.cross_entropy_loss(logits.float(), labels)
            metric.update({"ce": loss_cross_entropy.data})
            loss += loss_cross_entropy

            logits_v = self.visible_classifier(feature_representation[visible_mask])
            logits_i = self.infrared_classifier(feature_representation[infrared_mask])
            loss_id = 0
            loss_id += self.cross_entropy_loss(logits_v.float(), labels[visible_mask])
            loss_id += self.cross_entropy_loss(logits_i.float(), labels[infrared_mask])
            metric.update({"id": loss_id.data})
            loss += loss_id

            logits_m = torch.cat([logits_v, logits_i], 0).float()
            with torch.no_grad():
                # Momentum classifiers are EMA copies of modality-specific classifiers.
                # They provide cross-modality soft targets while avoiding direct gradient
                # feedback through the teacher branch in the same iteration.
                self.infrared_classifier_.weight.data = self.infrared_classifier_.weight.data * (1 - self.update_rate) + self.infrared_classifier.weight.data * self.update_rate
                self.visible_classifier_.weight.data = self.visible_classifier_.weight.data * (1 - self.update_rate) + self.visible_classifier.weight.data * self.update_rate

                logits_v_ = self.infrared_classifier_(feature_representation[visible_mask])
                logits_i_ = self.visible_classifier_(feature_representation[infrared_mask])
                logits_m_ = torch.cat([logits_v_, logits_i_], 0).float()
            loss_mc = self.cross_entropy_loss(logits_m, logits_m_.softmax(dim=1))
            metric.update({"mc": loss_mc.data})
            loss += loss_mc

            return loss, metric
        else:
            # Inference returns normalized retrieval features, not logits.
            feature_representation = self.bn_neck(feature_representation, modality_ids)
            return feature_representation
