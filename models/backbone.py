import torch
import torch.nn as nn
from torch.nn import init
from models.resnet import resnet50
import torch.nn.functional as F
import pdb

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

class Normalize(nn.Module):
    def __init__(self, power=2):
        super().__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1.0 / self.power)
        out = x.div(norm)
        return out

class SNA(torch.nn.Module):
    """Style normalization block that mixes original and normalized shallow features."""
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SNA, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "lambda=%f)" % self.e_lambda
        return s

    @staticmethod
    def get_module_name():
        return "sna"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

class CNL(nn.Module):
    """Cross-level non-local block for multi-level feature aggregation.

    x_h provides high-level semantic features and x_l provides lower-level
    spatial detail. The attention map is computed between projected high/low
    features, then low-level context is injected back into the high-level stream
    through a residual connection.
    """

    def __init__(self, high_dim, low_dim, flag=0):
        super(CNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim

        self.g = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
        if flag == 0:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=1, padding=0)
            self.W = nn.Sequential(
                nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(high_dim),
            )
        else:
            self.phi = nn.Conv2d(self.low_dim, self.low_dim, kernel_size=1, stride=2, padding=0)
            self.W = nn.Sequential(
                nn.Conv2d(self.low_dim, self.high_dim, kernel_size=1, stride=2, padding=0),
                nn.BatchNorm2d(self.high_dim),
            )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        phi_x = self.phi(x_l).view(B, self.low_dim, -1).permute(0, 2, 1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.view(B, self.low_dim, *x_l.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h

        return z

class PNL(nn.Module):
    """Pyramid non-local block for refining high-level features.

    Unlike CNL, this block computes attention in the high-level spatial domain
    and uses reduced low-level channels as values. The residual output keeps the
    semantic representation while reintroducing finer spatial evidence.
    """

    def __init__(self, high_dim, low_dim, reduc_ratio=2):
        super(PNL, self).__init__()
        self.high_dim = high_dim
        self.low_dim = low_dim
        self.reduc_ratio = reduc_ratio

        self.g = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(self.high_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(self.low_dim, self.low_dim // self.reduc_ratio, kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            nn.Conv2d(self.low_dim // self.reduc_ratio, self.high_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(high_dim),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

    def forward(self, x_h, x_l):
        B = x_h.size(0)
        g_x = self.g(x_l).view(B, self.low_dim, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x_h).view(B, self.low_dim, -1)
        theta_x = theta_x.permute(0, 2, 1)

        phi_x = self.phi(x_l).view(B, self.low_dim, -1)

        energy = torch.matmul(theta_x, phi_x)
        attention = energy / energy.size(-1)

        y = torch.matmul(attention, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(B, self.low_dim // self.reduc_ratio, *x_h.size()[2:])
        W_y = self.W(y)
        z = W_y + x_h
        return z

class MFA_block(nn.Module):
    """Multi-level feature aggregation block combining CNL and PNL."""

    def __init__(self, high_dim, low_dim, flag):
        super(MFA_block, self).__init__()

        self.CNL = CNL(high_dim, low_dim, flag)
        self.PNL = PNL(high_dim, low_dim)

    def forward(self, x, x0):
        z = self.CNL(x, x0)
        z = self.PNL(z, x0)
        return z

class MAM(nn.Module):
    """Modality-aware normalization for feature style control.

    Instance normalization suppresses modality-specific style such as color or
    thermal intensity. The learned channel mask decides per channel whether to
    preserve the raw feature or use the normalized feature.
    """

    def __init__(self, dim, r=16):
        super().__init__()

        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.IN = nn.InstanceNorm2d(dim, track_running_stats=False)

    def forward(self, x):
        pooled = F.avg_pool2d(x, x.size()[2:])
        mask = self.channel_attention(pooled)
        # The learned mask decides how much modality-specific style to normalize away.
        x = x * mask + self.IN(x) * (1 - mask)

        return x

class VisibleModule(nn.Module):
    def __init__(self):
        super().__init__()

        model_v = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x

class InfraredModule(nn.Module):
    def __init__(self):
        super().__init__()

        model_t = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)
        self.infrared = model_t

    def forward(self, x):
        x = self.infrared.conv1(x)
        x = self.infrared.bn1(x)
        x = self.infrared.relu(x)
        x = self.infrared.maxpool(x)
        return x

class SharedModule(nn.Module):
    def __init__(self):
        super().__init__()

        model_base = resnet50(pretrained=True, last_conv_stride=1, last_conv_dilation=1)

        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x

class Backbone(nn.Module):
    """Two-stream shallow stem plus shared ResNet trunk for VI-ReID features."""
    def __init__(
        self,
        num_classes,
        modality_attention=0,
    ):
        super().__init__()

        # Visible and infrared images use separate shallow stems before sharing deeper layers.
        self.visible_module = VisibleModule()
        self.infrared_module = InfraredModule()
        self.shared_module = SharedModule()

        layers = [3, 4, 6, 3]

        self.SNA = SNA()

        self.MFA1 = MFA_block(256, 64, 0)
        self.MFA2 = MFA_block(512, 256, 1)
        self.MFA3 = MFA_block(1024, 512, 1)

        self.modality_attention = modality_attention
        if self.modality_attention > 1:
            self.MAM3 = MAM(1024 if layers[3] > 2 else 256)
        if self.modality_attention > 0:
            self.MAM4 = MAM(2048 if layers[3] > 2 else 512)

        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(pool_dim, num_classes, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        dropout = 0
        self.dropout = dropout
        self.drop = nn.Dropout(self.dropout)
        out_planes = 64

        self.local_conv = nn.Conv2d(out_planes, 128, kernel_size=1, padding=0, bias=False)
        init.kaiming_normal_(self.local_conv.weight, mode="fan_out")
        self.feat_bn2d = nn.BatchNorm2d(128)
        init.constant_(self.feat_bn2d.weight, 1)
        init.constant_(self.feat_bn2d.bias, 0)
        self.max_pool_PCB = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, images, modality_ids):
        infrared_mask = modality_ids == 1
        visible_mask = modality_ids == 0

        infrared_features = self.infrared_module(images[infrared_mask])
        visible_features = self.visible_module(images[visible_mask])

        # Process modalities with separate shallow stems, then restore the original batch order.
        shallow_features = torch.empty(images.shape[0], infrared_features.shape[1], infrared_features.shape[2], infrared_features.shape[3], device=images.device, dtype=infrared_features.dtype)
        shallow_features[infrared_mask] = infrared_features
        shallow_features[visible_mask] = visible_features
        shared_features = self.SNA(shallow_features)

        layer1_features = self.shared_module.base.layer1(shared_features)
        shared_features = self.MFA1(layer1_features, shared_features)

        layer2_features = self.shared_module.base.layer2(shared_features)
        shared_features = self.MFA2(layer2_features, shared_features)

        layer3_features = self.shared_module.base.layer3(shared_features)
        shared_features = self.MFA3(layer3_features, shared_features)
        if self.modality_attention > 1:
            shared_features = self.MAM3(shared_features)

        feature_map = self.shared_module.base.layer4(shared_features)
        if self.modality_attention > 0:
            feature_map = self.MAM4(feature_map)
        feature_map = feature_map.float()

        return feature_map
