import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init, build_norm_layer
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence,
                                         build_attention,
                                         build_feedforward_network)
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.registry import (ATTENTION,TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.utils import deprecated_api_warning, ConfigDict
import warnings
import copy
from torch.nn import ModuleList
import torch.utils.checkpoint as cp

warnings.filterwarnings("ignore")

def get_global_pos(points, pc_range):
    points = points * (pc_range[3:6] - pc_range[0:3]) + pc_range[0:3]
    return points

@TRANSFORMER.register_module()
class Query_Transformer_RPL(BaseModule):
    def __init__(self,
                 decoder=None,
                 **kwargs):
        super(Query_Transformer_RPL, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def forward(self,
                query_weight,
                attn_masks=None,
                ):
        inter_states = self.decoder(
            query_weight = query_weight,
            attn_masks = attn_masks
            )

        return inter_states

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class Query_TransformerDecoder_RPL(TransformerLayerSequence):
    def __init__(self, 
                 *args, 
                 **kwargs):
        super(Query_TransformerDecoder_RPL, self).__init__(*args, **kwargs)

    def forward(self,
                query_weight,
                attn_masks=None,
                
                ):
        for lid, layer in enumerate(self.layers):
            query_weight = layer(
                query_weight,
                attn_masks
                
                )
        return query_weight

@TRANSFORMER_LAYER.register_module()
class Query_Transformer_DecoderLayer_self_RPL(BaseModule):
    def __init__(self,
                 attn_cfgs=None,
                 ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=256,
                     feedforward_channels=1024,
                     num_fcs=2,
                     ffn_drop=0.,
                     act_cfg=dict(type='ReLU', inplace=True),
                 ),
                 operation_order=None,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 batch_first=False,
                 with_cp=True,
                 **kwargs):
        super().__init__(init_cfg)

        self.batch_first = batch_first
        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = ModuleList()

        index = 0
        for operation_name in operation_order:
            if operation_name in ['self_attn', 'cross_attn']:
                if 'batch_first' in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]['batch_first']
                else:
                    attn_cfgs[index]['batch_first'] = self.batch_first
                attention = build_attention(attn_cfgs[index])
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims

        self.ffns = ModuleList()
        num_ffns = operation_order.count('ffn')
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = ConfigDict(ffn_cfgs)
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert len(ffn_cfgs) == num_ffns
        for ffn_index in range(num_ffns):
            if 'embed_dims' not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]['embed_dims'] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]['embed_dims'] == self.embed_dims
            self.ffns.append(
                build_feedforward_network(ffn_cfgs[ffn_index],
                                          dict(type='FFN')))

        self.norms = ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

        self.use_checkpoint = with_cp

    def _forward(self,
                query_weight,
                attn_masks=None,
                
                **kwargs):
        norm_index = 0
        attn_index = 0
        ffn_index = 0
        query_weight = query_weight.permute(1,0,2)
        
        identity = query_weight

        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                query_weight = self.attentions[attn_index](query_weight,
                                                         query_weight,
                                                         query_weight,
                                                         identity if self.pre_norm else None,
                                                         query_pos=None,
                                                         key_pos=None,
                                                         attn_mask=attn_masks[attn_index],
                                                         key_padding_mask=None,
                                                         **kwargs)
                attn_index += 1
                query_weight = query_weight + identity
                identity = query_weight

            elif layer == 'norm':
                
                query_weight = self.norms[norm_index](query_weight)
                identity = query_weight
                norm_index += 1

            elif layer == 'ffn':
                query_weight = self.ffns[ffn_index](
                    query_weight, identity if self.pre_norm else None)
                query_weight = query_weight + identity
                identity = query_weight
                ffn_index += 1
                
        query_weight = query_weight.permute(1,0,2)
        return query_weight

    def forward(self, 
                query_weight,
                attn_masks=None,
                ):
        if self.use_checkpoint and self.training:
            x = cp.checkpoint(
                self._forward, 
                query_weight,
                attn_masks
                )
        else:
            x = self._forward(
            query_weight,
            attn_masks
        )
        return x

