
import math
from operator import ipow
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head, conv
from lib.models.smmtrack.vit_decoupled import vit_base_patch16_224_smmtrack, vit_base_patch16_224_ce, vit_small_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
from mmcv.cnn import (ConvModule, build_norm_layer,
                      build_activation_layer)
from mmcv.runner import Sequential
import matplotlib.pyplot as plt

class SMMTrack(nn.Module):

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        hidden_dim = transformer.embed_dim
        self.backbone = transformer
        self.smmtrack_conv = conv(hidden_dim * 2, hidden_dim)  # Fuse RGB and T search regions, random initialized
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        # self.FAT = FAT(channels=768)
        # self.DTP = DTP(768,768,smooth=True)

    # def forward(self, template: torch.Tensor,
    #             search: torch.Tensor,
    #             ce_template_mask=None,
    #             ce_keep_rate=None,
    #             return_last_attn=False,
    #             ):
    #     x, aux_dict = self.backbone(z=template, x=search,
    #                                 ce_template_mask=ce_template_mask,
    #                                 ce_keep_rate=ce_keep_rate,
    #                                 return_last_attn=return_last_attn, )
    #     # Forward head
    #     feat_last = x
    #     if isinstance(x, list):
    #         feat_last = x[-1]
    #     out = self.forward_head(feat_last, None)
    #
    #     out.update(aux_dict)
    #     out['backbone_feat'] = x
    #     return out

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                hidden_state_ = None,
                info=None
                ):
        train = True
        if train is True:
            # out_dict = []
            # hidden_state = None
            # for i in range(len(search[0])):
            #     search_list = [search[0][i], search[1][i]]
            #     x, aux_dict = self.backbone(z=template, x=search_list,
            #                                 ce_template_mask=ce_template_mask,
            #                                 ce_keep_rate=ce_keep_rate,
            #                                 return_last_attn=return_last_attn, )
            #     # Forward head
            #     feat_last = x
            #     if isinstance(x, list):
            #         feat_last = x[-1]
            #     out, hidden_state= self.forward_head(feat_last, None, hidden_state)
            #
            #     out.update(aux_dict)
            #     out['backbone_feat'] = x
            #     out_dict.append(out)
            # return out_dict

            hidden_state = None
            # 这里只是修改了搜索区域的读取操作
            search_list = [search[0][0], search[1][0]]
            x, aux_dict = self.backbone(z=template, x=search_list,
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, )

            # Forward head
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]
            # out = self.forward_head(feat_last, None)
            # 沿用这个head
            out, hidden_state = self.forward_head(feat_last, None, hidden_state)

            out.update(aux_dict)
            out['backbone_feat'] = x
            return out
        else:
            # search_list = search
            x, aux_dict = self.backbone(z=template, x=search,
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn,
                                        info=info)
            # Forward head
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]
            out, hidden_state_ = self.forward_head(feat_last, None, hidden_state_, info)

            out.update(aux_dict)
            out['backbone_feat'] = x
            return out #, hidden_state_

    def forward_head(self, cat_feature, gt_score_map=None, hidden_state=None, info=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        size = "384_192"
        # size = "256_128"
        if size == "256_128":
            num_template_token = 320
            num_search_token = 256
        else:
            num_template_token = 720
            num_search_token = 576
        # encoder outputs for the visible and infrared search regions, both are (B, HW, C)
        enc_opt1 = cat_feature[:, num_template_token:num_template_token + num_search_token, :]#.transpose(1, 2).view(cat_feature.shape[0], 768, self.feat_sz_s, self.feat_sz_s)
        enc_opt2 = cat_feature[:, -num_search_token:, :]#.transpose(1, 2).view(cat_feature.shape[0], 768, self.feat_sz_s, self.feat_sz_s)
        enc_opt = torch.cat([enc_opt1, enc_opt2], dim=2)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        HW = int(HW/2)
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        # opt_feat = self.DTP(opt_feat)
        opt_feat = self.smmtrack_conv(opt_feat)
        # self.visualize_and_save_features(opt_feat, info['name'], info['number'], 'last', '0')
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)

            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out,hidden_state
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            # outputs_coord_new = outputs_coord.view(bs, 1, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out,hidden_state
        else:
            raise NotImplementedError

    # def forward_head(self, cat_feature, gt_score_map=None, hidden_state=None):
    #     """
    #     cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
    #     """
    #     size = "384_192"
    #     if size == "256_128":
    #         num_template_token = 320
    #         num_search_token = 256
    #     else:
    #         num_template_token = 720
    #         num_search_token = 576
    #     # encoder outputs for the visible and infrared search regions, both are (B, HW, C)
    #     enc_opt1 = cat_feature[:, num_template_token:num_template_token + num_search_token, :]#.transpose(1, 2).view(cat_feature.shape[0], 768, self.feat_sz_s, self.feat_sz_s)
    #     enc_opt2 = cat_feature[:, -num_search_token:, :]#.transpose(1, 2).view(cat_feature.shape[0], 768, self.feat_sz_s, self.feat_sz_s)
    #     enc_opt = torch.cat([enc_opt1, enc_opt2], dim=2)
    #     opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
    #     bs, Nq, C, HW = opt.size()
    #     HW = int(HW/2)
    #     opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
    #     if self.head_type == "CORNER":
    #         # run the corner head
    #         pred_box, score_map = self.box_head(opt_feat, True)
    #         outputs_coord = box_xyxy_to_cxcywh(pred_box)
    #         outputs_coord_new = outputs_coord.view(bs, Nq, 4)
    #
    #         out = {'pred_boxes': outputs_coord_new,
    #                'score_map': score_map,
    #                }
    #         return out,hidden_state
    #     elif self.head_type == "CENTER":
    #         # run the center head
    #         score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
    #         # outputs_coord = box_xyxy_to_cxcywh(bbox)
    #         outputs_coord = bbox
    #         outputs_coord_new = outputs_coord.view(bs, Nq, 4)
    #         # outputs_coord_new = outputs_coord.view(bs, 1, 4)
    #         out = {'pred_boxes': outputs_coord_new,
    #                'score_map': score_map_ctr,
    #                'size_map': size_map,
    #                'offset_map': offset_map}
    #         return out,hidden_state
    #     else:
    #         raise NotImplementedError


def build_smmtrack_track(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('SMMTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        print('Load pretrained model from: ' + pretrained)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_smmtrack':
        backbone = vit_base_patch16_224_smmtrack(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            )
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_small_patch16_224_ce':
        backbone = vit_small_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )

    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim
    patch_start_index = 1

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    # box_head = build_box_head(cfg, hidden_dim * 2)
    #
    # boxhead_weight_filter = lambda param_dict : {k.replace("box_head.",""):v for k,v in param_dict.items() if 'box_head' in k}


    box_head = build_box_head(cfg, hidden_dim)
    model = SMMTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'SMMTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        # other_pretrained_file = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE1)
        # other_model_checkpoint = torch.load(other_pretrained_file, map_location="cpu")
        # other_model_state_dict = other_model_checkpoint["net"]
        #
        # # 提取需要的权重
        # pos_embed_z = other_model_state_dict['backbone.pos_embed_z']
        # pos_embed_x = other_model_state_dict['backbone.pos_embed_x']

        pretrained_file = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained_file, map_location="cpu")


        # for k, v in list(checkpoint['net'].items()):
        #     if k == 'box_head.conv1_ctr.0.weight':
        #         checkpoint['net']['box_head.conv1_ctr.0.weight'] = torch.cat([v, v], 1)
        #     elif k == 'box_head.conv1_offset.0.weight':
        #         checkpoint['net']['box_head.conv1_offset.0.weight'] = torch.cat([v, v], 1)
        #     elif k == 'box_head.conv1_size.0.weight':
        #         checkpoint['net']['box_head.conv1_size.0.weight'] = torch.cat([v, v], 1)


        # del checkpoint['net']['backbone.pos_embed_z']
        # del checkpoint['net']['backbone.pos_embed_x']
        # checkpoint['net']['backbone.pos_embed_z'] = pos_embed_z
        # checkpoint['net']['backbone.pos_embed_x'] = pos_embed_x
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model

