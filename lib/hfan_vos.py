import torch
import torch.nn as nn
from torch import Tensor

import torch.nn.functional as F

from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      build_activation_layer, kaiming_init)
from mmcv.runner import Sequential

from torch import nn as nn

class SelfAttentionBlock(nn.Module):
    """General self-attention block/non-local block.

    Please refer to https://arxiv.org/abs/1706.03762 for details about key,
    query and value.

    Args:
        key_in_channels (int): Input channels of key feature.
        query_in_channels (int): Input channels of query feature.
        channels (int): Output channels of key/query transform.
        out_channels (int): Output channels.
        share_key_query (bool): Whether share projection weight between key
            and query projection.
        query_downsample (nn.Module): Query downsample module.
        key_downsample (nn.Module): Key downsample module.
        key_query_num_convs (int): Number of convs for key/query projection.
        value_num_convs (int): Number of convs for value projection.
        matmul_norm (bool): Whether normalize attention map with sqrt of
            channels
        with_out (bool): Whether use out projection.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict|None): Config of activation layers.
    """

    def __init__(self, key_in_channels, query_in_channels, channels,
                 out_channels, share_key_query, query_downsample,
                 key_downsample, key_query_num_convs, value_out_num_convs,
                 key_query_norm, value_out_norm, matmul_norm, with_out,
                 conv_cfg, norm_cfg, act_cfg):
        super(SelfAttentionBlock, self).__init__()
        if share_key_query:
            assert key_in_channels == query_in_channels
        self.key_in_channels = key_in_channels
        self.query_in_channels = query_in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.share_key_query = share_key_query
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.key_project = self.build_project(
            key_in_channels,
            channels,
            num_convs=key_query_num_convs,
            use_conv_module=key_query_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if share_key_query:
            self.query_project = self.key_project
        else:
            self.query_project = self.build_project(
                query_in_channels,
                channels,
                num_convs=key_query_num_convs,
                use_conv_module=key_query_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        self.value_project = self.build_project(
            key_in_channels,
            channels if with_out else out_channels,
            num_convs=value_out_num_convs,
            use_conv_module=value_out_norm,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if with_out:
            self.out_project = self.build_project(
                channels,
                out_channels,
                num_convs=value_out_num_convs,
                use_conv_module=value_out_norm,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.out_project = None

        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm

        self.init_weights()

    def init_weights(self):
        """Initialize weight of later layer."""
        if self.out_project is not None:
            if not isinstance(self.out_project, ConvModule):
                constant_init(self.out_project, 0)

    def build_project(self, in_channels, channels, num_convs, use_conv_module,
                      conv_cfg, norm_cfg, act_cfg):
        """Build projection layer for key/query/value/out."""
        if use_conv_module:
            convs = [
                ConvModule(
                    in_channels,
                    channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    ConvModule(
                        channels,
                        channels,
                        1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg))
        else:
            convs = [nn.Conv2d(in_channels, channels, 1)]
            for _ in range(num_convs - 1):
                convs.append(nn.Conv2d(channels, channels, 1))
        if len(convs) > 1:
            convs = nn.Sequential(*convs)
        else:
            convs = convs[0]
        return convs

    def forward(self, query_feats, key_feats):
        """Forward function."""
        batch_size = query_feats.size(0)
        query = self.query_project(query_feats)
        if self.query_downsample is not None:
            query = self.query_downsample(query)
        query = query.reshape(*query.shape[:2], -1)
        query = query.permute(0, 2, 1).contiguous()

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)
        key = key.reshape(*key.shape[:2], -1)
        value = value.reshape(*value.shape[:2], -1)
        value = value.permute(0, 2, 1).contiguous()

        sim_map = torch.matmul(query, key)
        if self.matmul_norm:
            sim_map = (self.channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])
        if self.out_project is not None:
            context = self.out_project(context)
        return context

# (CSS) Category-Specific Semantic
class CSS(nn.Module):
    def __init__(self, scale):
        super(CSS, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        # b, c, h, w = feats.size()
        # print(batch_size, num_classes, height, width)
        # print(b, c, h, w)
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context


# (POC) Primary Object Context
class POC(SelfAttentionBlock):
    def __init__(self, channels, inter_channels, scale, conv_cfg, norm_cfg,
                 act_cfg):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super(POC, self).__init__(
            key_in_channels=channels,
            query_in_channels=channels,
            channels=inter_channels,
            out_channels=channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            channels * 2,
            channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, query_feats, key_feats):
        """Forward function."""
        context = super(POC,
                        self).forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = resize(query_feats)

        return output


# baseline + FAM (Feature AlignMent)
# class FAM(nn.Module):
#     def __init__(self,
#                  channels=64,
#                  r=4,
#                  conv_cfg=None,
#                  norm_cfg=dict(type='BN'),
#                  act_cfg=dict(type='ReLU')
#                  ):
#         super(FAM, self).__init__()
#         inter_channels = int(channels // r)
#
#         self.poc = POC(
#             channels=channels,
#             inter_channels=inter_channels,
#             scale=1,
#             conv_cfg=conv_cfg,
#             norm_cfg=norm_cfg,
#             act_cfg=act_cfg)
#         self.css = CSS(scale=1)
#
#         self.object_pred = ConvModule(channels, 2, kernel_size=1)
#
#     def forward(self, im, fw):
#         im_pred = self.object_pred(im)
#         im_context = self.css(im, im_pred)
#         im_object_context = self.poc(im, im_context)
#         fw_object_context = self.poc(fw, im_context)
#
#         foc = im_object_context + fw_object_context
#
#         return foc
#
#
# # baseline + FAT (Feature AdaptaTion)
# class FAT(nn.Module):
#     def __init__(self,
#                  channels=64,
#                  r=4,
#                  conv_cfg=None,
#                  norm_cfg=dict(type='BN'),
#                  act_cfg=dict(type='ReLU')
#                  ):
#         super(FAT, self).__init__()
#         inter_channels = int(channels // r)
#
#         # channel attention
#         ca_conv1 = ConvModule(channels, inter_channels, kernel_size=1, stride=1, padding=0)
#         ca_bn1 = build_norm_layer(norm_cfg, inter_channels)[1]
#         ca_act1 = build_activation_layer(act_cfg)
#         ca_conv2 = ConvModule(inter_channels, channels, kernel_size=1, stride=1, padding=0)
#         ca_bn2 = build_norm_layer(norm_cfg, channels)[1]
#         ca_layers = [ca_conv1, ca_bn1, ca_act1, ca_conv2, ca_bn2]
#         self.ca_layers = Sequential(*ca_layers)
#         # pixel attention
#         ap = nn.AdaptiveAvgPool2d(1)
#         pa_conv1 = ConvModule(channels, inter_channels, kernel_size=1, stride=1, padding=0)
#         pa_bn1 = build_norm_layer(norm_cfg, inter_channels)[1]
#         pa_act1 = build_activation_layer(act_cfg)
#         pa_conv2 = ConvModule(inter_channels, channels, kernel_size=1, stride=1, padding=0)
#         pa_bn2 = build_norm_layer(norm_cfg, channels)[1]
#         pa_layers = [ap, pa_conv1, pa_bn1, pa_act1, pa_conv2, pa_bn2]
#         self.pa_layers = Sequential(*pa_layers)
#
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, im, fw):
#         f = im + fw
#         f_ca = self.ca_layers(f)
#         f_pa = self.pa_layers(f)
#         f_cp = f_ca + f_pa
#         w = self.sigmoid(f_cp)
#
#         f = 2 * im * w + 2 * fw * (1 - w)
#         return f


# HFAN (Hierarchical Feature Alignment Network)
class HFAN(nn.Module):
    def __init__(self,
                 channels=64,
                 r=4,
                 conv_cfg=None,
                 # norm_cfg=dict(type='SyncBN', requires_grad=True),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU')
                 ):
        super(HFAN, self).__init__()
        inter_channels = int(channels // r)

        # channel attention
        ca_conv1 = ConvModule(channels, inter_channels, kernel_size=1, stride=1, padding=0)
        ca_bn1 = build_norm_layer(norm_cfg, inter_channels)[1]
        ca_act1 = build_activation_layer(act_cfg)
        ca_conv2 = ConvModule(inter_channels, channels, kernel_size=1, stride=1, padding=0)
        ca_bn2 = build_norm_layer(norm_cfg, channels)[1]
        ca_layers = [ca_conv1, ca_bn1, ca_act1, ca_conv2, ca_bn2]
        self.ca_layers = Sequential(*ca_layers)
        # pixel attention
        ap = nn.AdaptiveAvgPool2d(1)
        pa_conv1 = ConvModule(channels, inter_channels, kernel_size=1, stride=1, padding=0)
        pa_bn1 = build_norm_layer(norm_cfg, inter_channels)[1]
        pa_act1 = build_activation_layer(act_cfg)
        pa_conv2 = ConvModule(inter_channels, channels, kernel_size=1, stride=1, padding=0)
        pa_bn2 = build_norm_layer(norm_cfg, channels)[1]
        pa_layers = [ap, pa_conv1, pa_bn1, pa_act1, pa_conv2, pa_bn2]
        self.pa_layers = Sequential(*pa_layers)

        self.sigmoid = nn.Sigmoid()

        self.poc = POC(
            channels=channels,
            inter_channels=inter_channels,
            scale=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            # norm_cfg=dict(type='SyncBN', requires_grad=True),
            act_cfg=act_cfg)
        self.css = CSS(scale=1)

        # self.object_pred = ConvModule(channels, 2, kernel_size=1)

    def forward(self, im, fw):
        # im_pred = self.object_pred(im)
        # im_context = self.css(im, im_pred)

        im_context = self.css(im, im)
        # im_context = self.css(im, fw)

        im_object_context = self.poc(im, im_context)
        fw_object_context = self.poc(fw, im_context)

        foc = im_object_context + fw_object_context
        foc_ca = self.ca_layers(foc)
        foc_pa = self.pa_layers(foc)
        foc_cp = foc_ca + foc_pa
        w = self.sigmoid(foc_cp)

        foc = 2 * im_object_context * w + 2 * fw_object_context * (1 - w)
        return foc


def split_batches(x: Tensor):
    """ Split a 2*B batch of images into two B images per batch,
    in order to adapt to MMSegmentation """

    assert x.ndim == 4, f'expect to have 4 dimensions, but got {x.ndim}'
    batch_size = x.shape[0] // 2
    x1 = x[0:batch_size, ...]
    x2 = x[batch_size:, ...]
    return x1, x2


def merge_batches(x1: Tensor, x2: Tensor):
    """ merge two batches each contains B images into a 2*B batch of images
    in order to adapt to MMSegmentation """

    assert x1.ndim == 4 and x2.ndim == 4, f'expect x1 and x2 to have 4 \
                dimensions, but got x1.dim: {x1.ndim}, x2.dim: {x2.ndim}'
    return torch.cat((x1, x2), dim=0)

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)