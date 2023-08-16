# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import imp
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.models.transformer import TransformerConfig
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise

from ..misc.load_krail import load_checkpoint_to_cpu

krail_map_list = ['krail', 'krail_ffn', 'krail_map', 'krail_map_avg', 'krail_map_last', 'krail_adaptive', 'krail_adaptive_final']

class TransformerEncoderLayerAdapter(nn.Module):

    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.encoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, cfg, return_fc=False):
        super().__init__()
        self.cfg = cfg
        self.return_fc = return_fc
        self.embed_dim = cfg.encoder.embed_dim
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size
        self.self_attn = self.build_self_attention(self.embed_dim, cfg)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = cfg.encoder.normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.encoder.ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            cfg.encoder.ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        self.num_heads = cfg.encoder.attention_heads
        self.load_to_BT = False
        self.ever_training = False
        # For BT, we need continuous mem
        self.in_proj_weight = torch.nn.Parameter(
            torch.zeros(
                self.self_attn.q_proj.weight.shape[0] * 3,
                self.self_attn.q_proj.weight.shape[1],
            )
        )
        self.in_proj_bias = torch.nn.Parameter(
            torch.zeros(self.self_attn.q_proj.bias.shape[0] * 3)
        )
        self.out_proj_weight = torch.nn.Parameter(
            torch.zeros(self.self_attn.out_proj.weight.shape)
        )
        self.out_proj_bias = torch.nn.Parameter(
            torch.zeros(self.self_attn.out_proj.bias.shape)
        )
        self.fc1_weight = torch.nn.Parameter(torch.zeros(self.fc1.weight.shape))
        self.fc1_bias = torch.nn.Parameter(torch.zeros(self.fc1.bias.shape))
        self.fc2_weight = torch.nn.Parameter(torch.zeros(self.fc2.weight.shape))
        self.fc2_bias = torch.nn.Parameter(torch.zeros(self.fc2.bias.shape))

        if (
            self.activation_fn is torch.nn.functional.relu
            or isinstance(self.activation_fn, torch.nn.ReLU)
            or self.activation_fn == "relu"
        ):
            self.activation_relu_or_gelu = 1
        elif (
            self.activation_fn is torch.nn.functional.gelu
            or isinstance(self.activation_fn, torch.nn.GELU)
            or self.activation_fn == "gelu"
        ):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        # Batch first can not be justified but needs user to make sure
        self.can_use_fastpath = (
            not self.normalize_before
            and self.activation_relu_or_gelu
            and (self.self_attn_layer_norm.eps == self.final_layer_norm.eps)
        )
        self.cfg_checkpoint_activations = self.cfg.checkpoint_activations
        # torch version check
        # make sure BT version is >=1.12.0
        self.BT_version = False
        if "fb" in torch.__version__:
            self.BT_version = True
        else:
            if "+" in torch.__version__:
                self.torch_version = torch.__version__.split("+")[0]
            else:
                self.torch_version = torch.__version__

            self.torch_version = self.torch_version.split(".")
            self.int_version = (
                int(self.torch_version[0]) * 1000
                + int(self.torch_version[1]) * 10
                + int(self.torch_version[2])
            )
            if len(self.torch_version) == 3:
                if self.int_version >= 1120:
                    self.BT_version = True
            elif len(self.torch_version) == 4:
                if self.int_version >= 1130:
                    self.BT_version = True

        # custom
        if cfg.freeze_all:
            for p in self.parameters():
                p.requires_grad = False
        
        if cfg.adapter_module != 'none':
            self.in_weight = nn.Linear(self.embed_dim, cfg.adapter_inner_dims, bias=True)
            if cfg.no_adapter_relu:
                self.adapter_relu=None
            else:
                self.adapter_relu=torch.nn.ReLU()
            self.out_weight = nn.Linear(cfg.adapter_inner_dims, self.embed_dim, bias=True)
            self.adapter_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        if cfg.adapter_module == 'parallel_gate':
            self.attn_gate = nn.Linear(self.embed_dim*2, self.embed_dim, bias=True)
            self.sigmod_gate = nn.Sigmoid()
            
        
    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self.load_to_BT = True

        old_name = prefix + "self_attn."
        q_proj_weight = state_dict[old_name + "q_proj.weight"]
        k_proj_weight = state_dict[old_name + "k_proj.weight"]
        v_proj_weight = state_dict[old_name + "v_proj.weight"]
        q_proj_bias = state_dict[old_name + "q_proj.bias"]
        k_proj_bias = state_dict[old_name + "k_proj.bias"]
        v_proj_bias = state_dict[old_name + "v_proj.bias"]

        new_name = prefix
        state_dict[new_name + "in_proj_weight"] = torch.cat(
            (q_proj_weight, k_proj_weight, v_proj_weight), dim=0
        )
        state_dict[new_name + "in_proj_bias"] = torch.cat(
            (q_proj_bias, k_proj_bias, v_proj_bias), dim=0
        )
        state_dict[new_name + "out_proj_weight"] = state_dict[
            old_name + "out_proj.weight"
        ]
        state_dict[new_name + "out_proj_bias"] = state_dict[old_name + "out_proj.bias"]
        state_dict[new_name + "fc1_weight"] = state_dict[prefix + "fc1.weight"]
        state_dict[new_name + "fc1_bias"] = state_dict[prefix + "fc1.bias"]
        state_dict[new_name + "fc2_weight"] = state_dict[prefix + "fc2.weight"]
        state_dict[new_name + "fc2_bias"] = state_dict[prefix + "fc2.bias"]
        
        

                
        if self.cfg.adapter_module == 'parallel_gate':
            if prefix+'attn_gate.weight' not in state_dict:
                state_dict[prefix + "attn_gate.weight"] =  self.attn_gate.weight
            if prefix+'attn_gate.bias' not in state_dict:
                state_dict[prefix + "attn_gate.bias"] =  self.attn_gate.bias

        if self.cfg.krail_model:
            if self.cfg.krail_module in krail_map_list:
                krail_state_dict = load_checkpoint_to_cpu(self.cfg.krail_model)
            else:
                krail_state_dict = None
        else:
            krail_state_dict = None

        if self.cfg.adapter_module != 'none' and krail_state_dict is not None and (self.cfg.krail_module == 'krail' or self.cfg.krail_module == 'krail_ffn'):
            print('----- load krail process done! Overwrite init adapters of '+prefix+' ! -----')
            if prefix+'in_weight.weight' not in state_dict:
                #state_dict[prefix + "in_weight.weight"] =  krail_state_dict['model'][prefix+'in_weight.weight']
                state_dict[new_name + "in_weight.weight"] =  krail_state_dict['model'][prefix+'fc1.weight']
                #print(state_dict[new_name + "in_weight.weight"])    
            if prefix+'in_weight.bias' not in state_dict:
                #state_dict[prefix + "in_weight.bias"] =  krail_state_dict['model'][prefix+'in_weight.bias']
                state_dict[new_name + "in_weight.bias"] =  krail_state_dict['model'][prefix+'fc1.bias']
            if prefix+'out_weight.weight' not in state_dict:
                #state_dict[prefix + "out_weight.weight"] =  krail_state_dict['model'][prefix+'out_weight.weight']
                state_dict[new_name + "out_weight.weight"] =  krail_state_dict['model'][prefix+'fc2.weight']
            if prefix+'out_weight.bias' not in state_dict:
                #state_dict[prefix + "out_weight.bias"] =  krail_state_dict['model'][prefix+'out_weight.bias']
                state_dict[new_name + "out_weight.bias"] =  krail_state_dict['model'][prefix+'fc2.bias']
            if prefix+'adapter_layer_norm.weight' not in state_dict:
                #state_dict[prefix + "adapter_layer_norm.weight"] =  krail_state_dict['model'][prefix+'adapter_layer_norm.weight']
                state_dict[new_name + "adapter_layer_norm.weight"] =  krail_state_dict['model'][prefix+'final_layer_norm.weight']
            if prefix+'adapter_layer_norm.bias' not in state_dict:
                #state_dict[prefix + "adapter_layer_norm.bias"] =  krail_state_dict['model'][prefix+'adapter_layer_norm.bias']
                state_dict[new_name + "adapter_layer_norm.bias"] =  krail_state_dict['model'][prefix+'final_layer_norm.bias']
        elif self.cfg.adapter_module != 'none' and krail_state_dict is not None and self.cfg.krail_module == 'krail_map':
            print('----- load krail process done! Overwrite init adapters of '+prefix+' with a map strategy! -----')
            # layer_num = int(prefix.split('.')[-2])
            # layer_num = layer_num + 18
            # map_name = 'encoder.layers.'+str(layer_num)+'.'
            # print('----- load krail process done! Overwrite init adapters of '+prefix+' with a map strategy on '+map_name+' ! -----')
            if prefix+'in_weight.weight' not in state_dict:
                #state_dict[prefix + "in_weight.weight"] =  krail_state_dict['model'][map_name+'fc1.weight']
                state_dict[new_name + "in_weight.weight"] =  krail_state_dict['model']['encoder.layers.11.fc1.weight']
                   
            if prefix+'in_weight.bias' not in state_dict:
                #state_dict[prefix + "in_weight.bias"] =  krail_state_dict['model'][map_name+'fc1.bias']
                state_dict[new_name + "in_weight.bias"] =  krail_state_dict['model']['encoder.layers.11.fc1.bias']
            if prefix+'out_weight.weight' not in state_dict:
                #state_dict[prefix + "out_weight.weight"] =  krail_state_dict['model'][map_name+'fc2.weight']
                state_dict[new_name + "out_weight.weight"] =  krail_state_dict['model']['encoder.layers.11.fc2.weight']
            if prefix+'out_weight.bias' not in state_dict:
                #state_dict[prefix + "out_weight.bias"] =  krail_state_dict['model'][map_name+'fc2.bias']
                state_dict[new_name + "out_weight.bias"] =  krail_state_dict['model']['encoder.layers.11.fc2.bias']
            if prefix+'adapter_layer_norm.weight' not in state_dict:
                #state_dict[prefix + "adapter_layer_norm.weight"] =  krail_state_dict['model'][map_name+'final_layer_norm.weight']
                state_dict[new_name + "adapter_layer_norm.weight"] =  krail_state_dict['model']['encoder.layers.11.final_layer_norm.weight']
            if prefix+'adapter_layer_norm.bias' not in state_dict:
                #state_dict[prefix + "adapter_layer_norm.bias"] =  krail_state_dict['model'][map_name+'final_layer_norm.bias']
                state_dict[new_name + "adapter_layer_norm.bias"] =  krail_state_dict['model']['encoder.layers.11.final_layer_norm.bias']
        elif self.cfg.adapter_module != 'none' and krail_state_dict is not None and self.cfg.krail_module == 'krail_map_last':
            layer_num = int(prefix.split('.')[-2])
            layer_num = layer_num + 6
            map_name = 'encoder.layers.'+str(layer_num)+'.'
            print('----- load krail process done! Overwrite init adapters of '+prefix+' with a map strategy on '+map_name+' ! -----')
            if prefix+'in_weight.weight' not in state_dict:
                state_dict[prefix + "in_weight.weight"] =  krail_state_dict['model'][map_name+'fc1.weight']       
            if prefix+'in_weight.bias' not in state_dict:
                state_dict[prefix + "in_weight.bias"] =  krail_state_dict['model'][map_name+'fc1.bias']
            if prefix+'out_weight.weight' not in state_dict:
                state_dict[prefix + "out_weight.weight"] =  krail_state_dict['model'][map_name+'fc2.weight']
            if prefix+'out_weight.bias' not in state_dict:
                state_dict[prefix + "out_weight.bias"] =  krail_state_dict['model'][map_name+'fc2.bias']
            if prefix+'adapter_layer_norm.weight' not in state_dict:
                state_dict[prefix + "adapter_layer_norm.weight"] =  krail_state_dict['model'][map_name+'final_layer_norm.weight']
            if prefix+'adapter_layer_norm.bias' not in state_dict:
                state_dict[prefix + "adapter_layer_norm.bias"] =  krail_state_dict['model'][map_name+'final_layer_norm.bias']
            
        elif self.cfg.adapter_module != 'none' and krail_state_dict is not None and self.cfg.krail_module == 'krail_map_avg':
            layer_num = int(prefix.split('.')[-2])
            split_num = 2
            print('----- load krail process done! Overwrite init adapters of '+prefix+' with a map strategy of average ! -----')
            if prefix+'in_weight.weight' not in state_dict:
                krail_tensor = krail_state_dict['model']['encoder.layers.'+str(layer_num*split_num)+'.fc1.weight']
                for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                    krail_tensor += krail_state_dict['model']['encoder.layers.'+str(i)+'.fc1.weight']
                krail_tensor = krail_tensor / split_num
                state_dict[prefix + "in_weight.weight"] =  krail_tensor
                #state_dict[new_name + "in_weight.weight"] =  krail_state_dict['model']['encoder.layers.23.fc1.weight']
                #print(state_dict[new_name + "in_weight.weight"])    
            if prefix+'in_weight.bias' not in state_dict:
                krail_tensor = krail_state_dict['model']['encoder.layers.'+str(layer_num*split_num)+'.fc1.bias']
                for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                    krail_tensor += krail_state_dict['model']['encoder.layers.'+str(i)+'.fc1.bias']
                krail_tensor = krail_tensor / split_num
                state_dict[prefix + "in_weight.bias"] =  krail_tensor
                #state_dict[new_name + "in_weight.bias"] =  krail_state_dict['model']['encoder.layers.23.fc1.bias']
            if prefix+'out_weight.weight' not in state_dict:
                krail_tensor = krail_state_dict['model']['encoder.layers.'+str(layer_num*split_num)+'.fc2.weight']
                for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                    krail_tensor += krail_state_dict['model']['encoder.layers.'+str(i)+'.fc2.weight']
                krail_tensor = krail_tensor / split_num
                state_dict[prefix + "out_weight.weight"] =  krail_tensor
                #state_dict[new_name + "out_weight.weight"] =  krail_state_dict['model']['encoder.layers.23.fc2.weight']
            if prefix+'out_weight.bias' not in state_dict:
                krail_tensor = krail_state_dict['model']['encoder.layers.'+str(layer_num*split_num)+'.fc2.bias']
                for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                    krail_tensor += krail_state_dict['model']['encoder.layers.'+str(i)+'.fc2.bias']
                krail_tensor = krail_tensor / split_num
                state_dict[prefix + "out_weight.bias"] =  krail_tensor
                #state_dict[new_name + "out_weight.bias"] =  krail_state_dict['model']['encoder.layers.23.fc2.bias']
            if prefix+'adapter_layer_norm.weight' not in state_dict:
                krail_tensor = krail_state_dict['model']['encoder.layers.'+str(layer_num*split_num)+'.final_layer_norm.weight']
                for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                    krail_tensor += krail_state_dict['model']['encoder.layers.'+str(i)+'.final_layer_norm.weight']
                krail_tensor = krail_tensor / split_num
                state_dict[prefix + "adapter_layer_norm.weight"] =  krail_tensor
                #state_dict[new_name + "adapter_layer_norm.weight"] =  krail_state_dict['model']['encoder.layers.23.final_layer_norm.weight']
            if prefix+'adapter_layer_norm.bias' not in state_dict:
                krail_tensor = krail_state_dict['model']['encoder.layers.'+str(layer_num*split_num)+'.final_layer_norm.bias']
                for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                    krail_tensor += krail_state_dict['model']['encoder.layers.'+str(i)+'.final_layer_norm.bias']
                krail_tensor = krail_tensor / split_num
                state_dict[prefix + "adapter_layer_norm.bias"] =  krail_tensor
                #state_dict[new_name + "adapter_layer_norm.bias"] =  krail_state_dict['model']['encoder.layers.23.final_layer_norm.bias']
        
        elif self.cfg.adapter_module != 'none' and krail_state_dict is not None and self.cfg.krail_module == 'krail_adaptive':
            layer_num = int(prefix.split('.')[-2])
            print('----- load krail process done! Overwrite init adapters of '+prefix+' with a map strategy of PLM_adapter ! -----')
            split_num = 2
            if prefix+'in_weight.weight' not in state_dict:
                # krail_tensor = krail_state_dict['model']['decoder.layers.'+str(layer_num*split_num)+'.fc1.weight']
                # for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                #     krail_tensor += krail_state_dict['model']['decoder.layers.'+str(i)+'.fc1.weight']
                # krail_tensor = krail_tensor / split_num
                # state_dict[prefix + "in_weight.weight"] =  krail_tensor
                state_dict[prefix + "in_weight.weight"] =  krail_state_dict['model'][prefix+'in_weight.weight']
                #print(state_dict[new_name + "in_weight.weight"])    
            if prefix+'in_weight.bias' not in state_dict:
                # krail_tensor = krail_state_dict['model']['decoder.layers.'+str(layer_num*split_num)+'.fc1.bias']
                # for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                #     krail_tensor += krail_state_dict['model']['decoder.layers.'+str(i)+'.fc1.bias']
                # krail_tensor = krail_tensor / split_num
                # state_dict[prefix + "in_weight.bias"] =  krail_tensor
                state_dict[prefix + "in_weight.bias"] =  krail_state_dict['model'][prefix+'in_weight.bias']
                # #state_dict[new_name + "in_weight.bias"] =  krail_state_dict['model']['encoder.layers.23.fc1.bias']
            if prefix+'out_weight.weight' not in state_dict:
                # krail_tensor = krail_state_dict['model']['decoder.layers.'+str(layer_num*split_num)+'.fc2.weight']
                # for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                #     krail_tensor += krail_state_dict['model']['decoder.layers.'+str(i)+'.fc2.weight']
                # krail_tensor = krail_tensor / split_num
                # state_dict[prefix + "out_weight.weight"] =  krail_tensor
                state_dict[prefix + "out_weight.weight"] =  krail_state_dict['model'][prefix+'out_weight.weight']
                #state_dict[new_name + "out_weight.weight"] =  krail_state_dict['model']['encoder.layers.23.fc2.weight']
            if prefix+'out_weight.bias' not in state_dict:
                # krail_tensor = krail_state_dict['model']['decoder.layers.'+str(layer_num*split_num)+'.fc2.bias']
                # for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                #     krail_tensor += krail_state_dict['model']['decoder.layers.'+str(i)+'.fc2.bias']
                # krail_tensor = krail_tensor / split_num
                # state_dict[prefix + "out_weight.bias"] =  krail_tensor
                state_dict[prefix + "out_weight.bias"] =  krail_state_dict['model'][prefix+'out_weight.bias']
                # #state_dict[new_name + "out_weight.bias"] =  krail_state_dict['model']['encoder.layers.23.fc2.bias']
            if prefix+'adapter_layer_norm.weight' not in state_dict:
                # krail_tensor = krail_state_dict['model']['decoder.layers.'+str(layer_num*split_num)+'.final_layer_norm.weight']
                # for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                #     krail_tensor += krail_state_dict['model']['decoder.layers.'+str(i)+'.final_layer_norm.weight']
                # krail_tensor = krail_tensor / split_num
                # state_dict[prefix + "adapter_layer_norm.weight"] =  krail_tensor
                state_dict[prefix + "adapter_layer_norm.weight"] =  krail_state_dict['model'][prefix+'adapter_layer_norm.weight']
                #state_dict[new_name + "adapter_layer_norm.weight"] =  krail_state_dict['model']['encoder.layers.23.final_layer_norm.weight']
            if prefix+'adapter_layer_norm.bias' not in state_dict:
                # krail_tensor = krail_state_dict['model']['decoder.layers.'+str(layer_num*split_num)+'.final_layer_norm.bias']
                # for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                #     krail_tensor += krail_state_dict['model']['decoder.layers.'+str(i)+'.final_layer_norm.bias']
                # krail_tensor = krail_tensor / split_num
                # state_dict[prefix + "adapter_layer_norm.bias"] =  krail_tensor
                state_dict[prefix + "adapter_layer_norm.bias"] =  krail_state_dict['model'][prefix+'adapter_layer_norm.bias']
                # #state_dict[new_name + "adapter_layer_norm.bias"] =  krail_state_dict['model']['encoder.layers.23.final_layer_norm.bias']
        elif self.cfg.adapter_module != 'none' and krail_state_dict is not None and self.cfg.krail_module == 'krail_adaptive_final':
            layer_num = int(prefix.split('.')[-2])
            print('----- load krail process done! Overwrite init adapters of '+prefix+' with a map strategy of PLM_adapter_final ! -----')
            split_num = 2
            if prefix+'in_weight.weight' not in state_dict:
                state_dict[prefix + "in_weight.weight"] =  krail_state_dict['model']['encoder.layers.23.in_weight.weight'] 
            if prefix+'in_weight.bias' not in state_dict:
                state_dict[prefix + "in_weight.bias"] =  krail_state_dict['model']['encoder.layers.23.in_weight.bias']
            if prefix+'out_weight.weight' not in state_dict:
                state_dict[prefix + "out_weight.weight"] =  krail_state_dict['model']['encoder.layers.23.out_weight.weight']
            if prefix+'out_weight.bias' not in state_dict:
                state_dict[prefix + "out_weight.bias"] =  krail_state_dict['model']['encoder.layers.23.out_weight.bias']
            if prefix+'adapter_layer_norm.weight' not in state_dict:
               state_dict[prefix + "adapter_layer_norm.weight"] =  krail_state_dict['model']['encoder.layers.23.adapter_layer_norm.weight']
            if prefix+'adapter_layer_norm.bias' not in state_dict:
                state_dict[prefix + "adapter_layer_norm.bias"] =  krail_state_dict['model']['encoder.layers.23.adapter_layer_norm.bias']
                
        elif self.cfg.adapter_module != 'none' and krail_state_dict is None:
            if prefix+'in_weight.weight' not in state_dict:
                state_dict[new_name + "in_weight.weight"] =  self.in_weight.weight    
            if prefix+'in_weight.bias' not in state_dict:
                state_dict[new_name + "in_weight.bias"] =  self.in_weight.bias
            if prefix+'out_weight.weight' not in state_dict:
                state_dict[new_name + "out_weight.weight"] =  self.out_weight.weight
            if prefix+'out_weight.bias' not in state_dict:
                state_dict[new_name + "out_weight.bias"] =  self.out_weight.bias
            if prefix+'adapter_layer_norm.weight' not in state_dict:
                state_dict[new_name + "adapter_layer_norm.weight"] =  self.adapter_layer_norm.weight
            if prefix+'adapter_layer_norm.bias' not in state_dict:
                state_dict[new_name + "adapter_layer_norm.bias"] =  self.adapter_layer_norm.bias
        
        super(TransformerEncoderLayerAdapter, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def _get_fc_rank(self, remove_num: int) -> List[int]:
        f1_filter_param = []
        for i in range(self.fc1.out_features):
            f1_filter_param.append(
                torch.sum(torch.abs(self.fc1.weight[i]))
                + torch.sum(torch.abs(self.fc2.weight[:, i]))
                + torch.abs(self.fc1.bias[i])
            )
        return sorted(
            range(len(f1_filter_param)), key=lambda k: f1_filter_param[k], reverse=False
        )[0:remove_num]

    def _prune_fc_layer(self, remove_index: List[int]):
        new_fc1_weight = []
        new_fc1_bias = []
        for i in range(self.fc1.out_features):
            if i not in remove_index:
                new_fc1_weight.append(self.fc1.weight[i])
                new_fc1_bias.append(self.fc1.bias[i])

        new_fc1_weight = torch.stack(new_fc1_weight).detach()
        new_fc1_weight.requires_grad = True

        new_fc1_bias = torch.stack(new_fc1_bias).detach()
        new_fc1_bias.requires_grad = True

        self.fc1 = quant_noise(
            nn.Linear(self.fc1.in_features, self.fc1.out_features - len(remove_index)),
            p=self.quant_noise,
            block_size=self.quant_noise_block_size,
        )
        self.fc1.weight = torch.nn.Parameter(new_fc1_weight)
        self.fc1.bias = torch.nn.Parameter(new_fc1_bias)

        new_fc2_weight = []
        new_fc2_bias = []
        for i in range(self.fc2.in_features):
            if i not in remove_index:
                new_fc2_weight.append(self.fc2.weight[:, i])
        new_fc2_bias = self.fc2.bias.detach()

        new_fc2_weight = torch.stack(new_fc2_weight, dim=-1).detach()
        new_fc2_weight.requires_grad = True

        new_fc2_bias = self.fc2.bias.detach()
        new_fc2_bias.requires_grad = True

        self.fc2 = quant_noise(
            nn.Linear(self.fc2.in_features - len(remove_index), self.fc2.out_features),
            p=self.quant_noise,
            block_size=self.quant_noise_block_size,
        )
        self.fc2.weight = torch.nn.Parameter(new_fc2_weight)
        self.fc2.bias = torch.nn.Parameter(new_fc2_bias)

    def build_self_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.encoder.attention_heads,
            dropout=cfg.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=cfg.encoder.xformers_att_config,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters

        if self.training:
            self.ever_training = True
            
        if (
            self.BT_version
            and x.dim() == 3
            and self.load_to_BT
            and not self.return_fc
            and self.can_use_fastpath
            and not self.training
            and not self.ever_training
            and not self.cfg_checkpoint_activations
        ):
            # assume is Batch first and nested tensor
            output = torch._transformer_encoder_layer_fwd(
                x,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj_weight,
                self.out_proj_bias,
                self.activation_relu_or_gelu == 2,
                False,  # norm_first, currently not supported
                self.self_attn_layer_norm.eps,
                self.self_attn_layer_norm.weight,
                self.self_attn_layer_norm.bias,
                self.final_layer_norm.weight,
                self.final_layer_norm.bias,
                self.fc1_weight,
                self.fc1_bias,
                self.fc2_weight,
                self.fc2_bias,
                encoder_padding_mask if encoder_padding_mask is not None else attn_mask,
            )
            return output

        else:
            if attn_mask is not None:
                attn_mask = attn_mask.masked_fill(
                    attn_mask.to(torch.bool), -1e8 if x.dtype == torch.float32 else -1e4
                )

            residual = x
            if self.normalize_before:
                x = self.self_attn_layer_norm(x)
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.self_attn_layer_norm(x)

            # compute FFN
            # parallel adapter
            
            residual = x

            if self.cfg.adapter_module == 'parallel_norm':
                if self.normalize_before:
                    x = self.final_layer_norm(x)
                    px = self.adapter_layer_norm(x)
            else:
                if self.normalize_before:
                    x = self.final_layer_norm(x)

            if self.cfg.adapter_module == 'parallel' or self.cfg.adapter_module == 'parallel_dp' or self.cfg.adapter_module == 'parallel_norm':
                if self.cfg.adapter_module == 'parallel_norm':
                    px = self.in_weight(px)
                else:
                    px = self.in_weight(x)

                if self.cfg.no_adapter_relu:
                    pass
                else:
                    px = self.adapter_relu(px)
                px = self.out_weight(px)
                if self.cfg.adapter_module == 'parallel_dp':
                    pass
                else:
                    px = self.dropout_module(px)

                x = self.activation_fn(self.fc1(x))
                x = self.activation_dropout_module(x)
                x = self.fc2(x)
                x = x + px
            elif self.cfg.adapter_module == 'parallel_gate':
                px = self.in_weight(x)
                if self.cfg.no_adapter_relu:
                    pass
                else:
                    px = self.adapter_relu(px)
                px = self.out_weight(px)
                px = self.dropout_module(px)

                gx = torch.cat([x, px], dim=-1)

                gx = self.sigmod_gate(self.attn_gate(gx))

                x = x * gx + px * (1-gx)
            else:

                x = self.activation_fn(self.fc1(x))
                x = self.activation_dropout_module(x)
                x = self.fc2(x)


            fc_result = x                

            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)
           
           
            # serial adapter
            if self.cfg.adapter_module == 'serial' or self.cfg.adapter_module == 'serial_single':

                residual = x
                if self.normalize_before:
                    x = self.adapter_layer_norm(x)

                x = self.in_weight(x)
                if self.cfg.no_adapter_relu:
                    pass
                else:
                    x = self.adapter_relu(x)
                x = self.out_weight(x)
                x = self.dropout_module(x)
                x = self.residual_connection(x, residual)
                if not self.normalize_before:
                    x = self.final_layer_norm(x)

            if self.return_fc and not torch.jit.is_scripting():
                return x, fc_result


            return x


# backward compatible with the legacy argparse format
class TransformerEncoderLayer(TransformerEncoderLayerAdapter):
    def __init__(self, args):
        super().__init__(TransformerConfig.from_namespace(args))
        self.args = args

    def build_self_attention(self, embed_dim, args):
        return super().build_self_attention(
            embed_dim, TransformerConfig.from_namespace(args)
        )


class TransformerDecoderLayerAdapter(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *cfg.decoder.normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, cfg, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.decoder.embed_dim
        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = cfg.quant_noise.pq
        self.quant_noise_block_size = cfg.quant_noise.pq_block_size

        self.cross_self_attention = cfg.cross_self_attention

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            cfg,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.attn_ln = (
            LayerNorm(self.embed_dim)
            if utils.safe_getattr(cfg, "scale_attn", False)
            else None
        )
        self.nh = self.self_attn.num_heads
        self.head_dim = self.self_attn.head_dim
        scale_heads = utils.safe_getattr(cfg, "scale_heads", False)
        self.c_attn = (
            nn.Parameter(torch.ones((self.nh,)), requires_grad=True)
            if scale_heads
            else None
        )

        self.activation_fn = utils.get_activation_fn(activation=cfg.activation_fn)
        activation_dropout_p = cfg.activation_dropout
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use cfg.relu_dropout
            activation_dropout_p = cfg.relu_dropout or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = cfg.decoder.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, cfg)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)

        self.ffn_layernorm = (
            LayerNorm(cfg.decoder.ffn_embed_dim)
            if utils.safe_getattr(cfg, "scale_fc", False)
            else None
        )
        self.w_resid = (
            nn.Parameter(
                torch.ones(
                    self.embed_dim,
                ),
                requires_grad=True,
            )
            if utils.safe_getattr(cfg, "scale_resids", False)
            else None
        )

        self.fc1 = self.build_fc1(
            self.embed_dim,
            cfg.decoder.ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            cfg.decoder.ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        self.need_attn = True

        self.onnx_trace = False
        
        # custom
        if cfg.freeze_all:
            for p in self.parameters():
                p.requires_grad = False
        
        #初始化decoder的adapter
        if cfg.adapter_module != 'none':
            self.in_weight = nn.Linear(self.embed_dim, cfg.adapter_inner_dims, bias=True)
            if cfg.no_adapter_relu:
                self.adapter_relu=None
            else:
                self.adapter_relu=torch.nn.ReLU()
            self.out_weight = nn.Linear(cfg.adapter_inner_dims, self.embed_dim, bias=True)
            self.adapter_layer_norm = LayerNorm(self.embed_dim, export=cfg.export)
        if cfg.adapter_module == 'parallel_gate':
            self.attn_gate = nn.Linear(self.embed_dim*2, self.embed_dim, bias=True)
            self.sigmod_gate = nn.Sigmoid()            

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):        
        
        if self.cfg.adapter_module == 'parallel_gate':
            if prefix+'attn_gate.weight' not in state_dict:
                state_dict[prefix + "attn_gate.weight"] =  self.attn_gate.weight
            if prefix+'attn_gate.bias' not in state_dict:
                state_dict[prefix + "attn_gate.bias"] =  self.attn_gate.bias
                
        if self.cfg.krail_model:
            if self.cfg.krail_module in krail_map_list:
                krail_state_dict = load_checkpoint_to_cpu(self.cfg.krail_model)
            else:
                krail_state_dict = None
        else:
            krail_state_dict = None
            
        if self.cfg.adapter_module != 'none' and krail_state_dict is not None and (self.cfg.krail_module == 'krail' or self.cfg.krail_module == 'krail_ffn'):
            print('----- load krail process done! Overwrite init adapters of '+prefix+' ! -----')
            if prefix+'in_weight.weight' not in state_dict:
                # print(state_dict[prefix + "in_weight.weight"])
                #state_dict[prefix + "in_weight.weight"] =  krail_state_dict['model'][prefix+'in_weight.weight']
                state_dict[prefix + "in_weight.weight"] =  krail_state_dict['model'][prefix+'fc1.weight']    
            if prefix+'in_weight.bias' not in state_dict:
                #state_dict[prefix + "in_weight.bias"] =  krail_state_dict['model'][prefix+'in_weight.bias']
                state_dict[prefix + "in_weight.bias"] =  krail_state_dict['model'][prefix+'fc1.bias']
            if prefix+'out_weight.weight' not in state_dict:
                #state_dict[prefix + "out_weight.weight"] =  krail_state_dict['model'][prefix+'out_weight.weight']
                state_dict[prefix + "out_weight.weight"] =  krail_state_dict['model'][prefix+'fc2.weight']
            if prefix+'out_weight.bias' not in state_dict:
                #state_dict[prefix + "out_weight.bias"] =  krail_state_dict['model'][prefix+'out_weight.bias']
                state_dict[prefix + "out_weight.bias"] =  krail_state_dict['model'][prefix+'fc2.bias']
            if prefix+'adapter_layer_norm.weight' not in state_dict:
                #state_dict[prefix + "adapter_layer_norm.weight"] =  krail_state_dict['model'][prefix+'adapter_layer_norm.weight']
                state_dict[prefix + "adapter_layer_norm.weight"] =  krail_state_dict['model'][prefix+'final_layer_norm.weight']
            if prefix+'adapter_layer_norm.bias' not in state_dict:
                #state_dict[prefix + "adapter_layer_norm.bias"] =  krail_state_dict['model'][prefix+'adapter_layer_norm.bias']
                state_dict[prefix + "adapter_layer_norm.bias"] =  krail_state_dict['model'][prefix+'final_layer_norm.bias']
        elif self.cfg.adapter_module != 'none' and krail_state_dict is not None and self.cfg.krail_module == 'krail_map':
            print('----- load krail process done! Overwrite init adapters of '+prefix+' with a map strategy! -----')
            # layer_num = int(prefix.split('.')[-2])
            # layer_num = layer_num + 18
            # map_name = 'decoder.layers.'+str(layer_num)+'.'
            # print('----- load krail process done! Overwrite init adapters of '+prefix+' with a map strategy on '+map_name+' ! -----')
            if prefix+'in_weight.weight' not in state_dict:
                #state_dict[prefix + "in_weight.weight"] =  krail_state_dict['model'][map_name+'fc1.weight']
                state_dict[prefix + "in_weight.weight"] =  krail_state_dict['model']['decoder.layers.11.fc1.weight']    
            if prefix+'in_weight.bias' not in state_dict:
                #state_dict[prefix + "in_weight.bias"] =  krail_state_dict['model'][map_name+'fc1.bias']
                state_dict[prefix + "in_weight.bias"] =  krail_state_dict['model']['decoder.layers.11.fc1.bias']
            if prefix+'out_weight.weight' not in state_dict:
                #state_dict[prefix + "out_weight.weight"] =  krail_state_dict['model'][map_name+'fc2.weight']
                state_dict[prefix + "out_weight.weight"] =  krail_state_dict['model']['decoder.layers.11.fc2.weight']
            if prefix+'out_weight.bias' not in state_dict:
                #state_dict[prefix + "out_weight.bias"] =  krail_state_dict['model'][map_name+'fc2.bias']
                state_dict[prefix + "out_weight.bias"] =  krail_state_dict['model']['decoder.layers.11.fc2.bias']
            if prefix+'adapter_layer_norm.weight' not in state_dict:
                #state_dict[prefix + "adapter_layer_norm.weight"] =  krail_state_dict['model'][map_name+'final_layer_norm.weight']
                state_dict[prefix + "adapter_layer_norm.weight"] =  krail_state_dict['model']['decoder.layers.11.final_layer_norm.weight']
            if prefix+'adapter_layer_norm.bias' not in state_dict:
                #state_dict[prefix + "adapter_layer_norm.bias"] =  krail_state_dict['model'][map_name+'final_layer_norm.bias']
                state_dict[prefix + "adapter_layer_norm.bias"] =  krail_state_dict['model']['decoder.layers.11.final_layer_norm.bias']
        elif self.cfg.adapter_module != 'none' and krail_state_dict is not None and self.cfg.krail_module == 'krail_map_last':
            layer_num = int(prefix.split('.')[-2])
            layer_num = layer_num + 6
            map_name = 'decoder.layers.'+str(layer_num)+'.'
            print('----- load krail process done! Overwrite init adapters of '+prefix+' with a map strategy on '+map_name+' ! -----')
            if prefix+'in_weight.weight' not in state_dict:
                state_dict[prefix + "in_weight.weight"] =  krail_state_dict['model'][map_name+'fc1.weight']  
            if prefix+'in_weight.bias' not in state_dict:
                state_dict[prefix + "in_weight.bias"] =  krail_state_dict['model'][map_name+'fc1.bias']
            if prefix+'out_weight.weight' not in state_dict:
                state_dict[prefix + "out_weight.weight"] =  krail_state_dict['model'][map_name+'fc2.weight']
            if prefix+'out_weight.bias' not in state_dict:
                state_dict[prefix + "out_weight.bias"] =  krail_state_dict['model'][map_name+'fc2.bias']
            if prefix+'adapter_layer_norm.weight' not in state_dict:
                state_dict[prefix + "adapter_layer_norm.weight"] =  krail_state_dict['model'][map_name+'final_layer_norm.weight']
            if prefix+'adapter_layer_norm.bias' not in state_dict:
                state_dict[prefix + "adapter_layer_norm.bias"] =  krail_state_dict['model'][map_name+'final_layer_norm.bias']
        elif self.cfg.adapter_module != 'none' and krail_state_dict is not None and self.cfg.krail_module == 'krail_map_avg':
            layer_num = int(prefix.split('.')[-2])
            print('----- load krail process done! Overwrite init adapters of '+prefix+' with a map strategy of average ! -----')
            split_num = 2
            if prefix+'in_weight.weight' not in state_dict:
                krail_tensor = krail_state_dict['model']['decoder.layers.'+str(layer_num*split_num)+'.fc1.weight']
                for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                    krail_tensor += krail_state_dict['model']['decoder.layers.'+str(i)+'.fc1.weight']
                krail_tensor = krail_tensor / split_num
                state_dict[prefix + "in_weight.weight"] =  krail_tensor
                #state_dict[new_name + "in_weight.weight"] =  krail_state_dict['model']['encoder.layers.23.fc1.weight']
                #print(state_dict[new_name + "in_weight.weight"])    
            if prefix+'in_weight.bias' not in state_dict:
                krail_tensor = krail_state_dict['model']['decoder.layers.'+str(layer_num*split_num)+'.fc1.bias']
                for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                    krail_tensor += krail_state_dict['model']['decoder.layers.'+str(i)+'.fc1.bias']
                krail_tensor = krail_tensor / split_num
                state_dict[prefix + "in_weight.bias"] =  krail_tensor
                #state_dict[new_name + "in_weight.bias"] =  krail_state_dict['model']['encoder.layers.23.fc1.bias']
            if prefix+'out_weight.weight' not in state_dict:
                krail_tensor = krail_state_dict['model']['decoder.layers.'+str(layer_num*split_num)+'.fc2.weight']
                for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                    krail_tensor += krail_state_dict['model']['decoder.layers.'+str(i)+'.fc2.weight']
                krail_tensor = krail_tensor / split_num
                state_dict[prefix + "out_weight.weight"] =  krail_tensor
                #state_dict[new_name + "out_weight.weight"] =  krail_state_dict['model']['encoder.layers.23.fc2.weight']
            if prefix+'out_weight.bias' not in state_dict:
                krail_tensor = krail_state_dict['model']['decoder.layers.'+str(layer_num*split_num)+'.fc2.bias']
                for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                    krail_tensor += krail_state_dict['model']['decoder.layers.'+str(i)+'.fc2.bias']
                krail_tensor = krail_tensor / split_num
                state_dict[prefix + "out_weight.bias"] =  krail_tensor
                #state_dict[new_name + "out_weight.bias"] =  krail_state_dict['model']['encoder.layers.23.fc2.bias']
            if prefix+'adapter_layer_norm.weight' not in state_dict:
                krail_tensor = krail_state_dict['model']['decoder.layers.'+str(layer_num*split_num)+'.final_layer_norm.weight']
                for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                    krail_tensor += krail_state_dict['model']['decoder.layers.'+str(i)+'.final_layer_norm.weight']
                krail_tensor = krail_tensor / split_num
                state_dict[prefix + "adapter_layer_norm.weight"] =  krail_tensor
                #state_dict[new_name + "adapter_layer_norm.weight"] =  krail_state_dict['model']['encoder.layers.23.final_layer_norm.weight']
            if prefix+'adapter_layer_norm.bias' not in state_dict:
                krail_tensor = krail_state_dict['model']['decoder.layers.'+str(layer_num*split_num)+'.final_layer_norm.bias']
                for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                    krail_tensor += krail_state_dict['model']['decoder.layers.'+str(i)+'.final_layer_norm.bias']
                krail_tensor = krail_tensor / split_num
                state_dict[prefix + "adapter_layer_norm.bias"] =  krail_tensor
                #state_dict[new_name + "adapter_layer_norm.bias"] =  krail_state_dict['model']['encoder.layers.23.final_layer_norm.bias']
        elif self.cfg.adapter_module != 'none' and krail_state_dict is not None and self.cfg.krail_module == 'krail_adaptive':
            layer_num = int(prefix.split('.')[-2])
            print('----- load krail process done! Overwrite init adapters of '+prefix+' with a map strategy of PLM_adapter ! -----')
            split_num = 2
            if prefix+'in_weight.weight' not in state_dict:
                # krail_tensor = krail_state_dict['model']['decoder.layers.'+str(layer_num*split_num)+'.fc1.weight']
                # for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                #     krail_tensor += krail_state_dict['model']['decoder.layers.'+str(i)+'.fc1.weight']
                # krail_tensor = krail_tensor / split_num
                # state_dict[prefix + "in_weight.weight"] =  krail_tensor
                state_dict[prefix + "in_weight.weight"] =  krail_state_dict['model'][prefix+'in_weight.weight']
                #print(state_dict[new_name + "in_weight.weight"])    
            if prefix+'in_weight.bias' not in state_dict:
                # krail_tensor = krail_state_dict['model']['decoder.layers.'+str(layer_num*split_num)+'.fc1.bias']
                # for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                #     krail_tensor += krail_state_dict['model']['decoder.layers.'+str(i)+'.fc1.bias']
                # krail_tensor = krail_tensor / split_num
                # state_dict[prefix + "in_weight.bias"] =  krail_tensor
                state_dict[prefix + "in_weight.bias"] =  krail_state_dict['model'][prefix+'in_weight.bias']
                # #state_dict[new_name + "in_weight.bias"] =  krail_state_dict['model']['encoder.layers.23.fc1.bias']
            if prefix+'out_weight.weight' not in state_dict:
                # krail_tensor = krail_state_dict['model']['decoder.layers.'+str(layer_num*split_num)+'.fc2.weight']
                # for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                #     krail_tensor += krail_state_dict['model']['decoder.layers.'+str(i)+'.fc2.weight']
                # krail_tensor = krail_tensor / split_num
                # state_dict[prefix + "out_weight.weight"] =  krail_tensor
                state_dict[prefix + "out_weight.weight"] =  krail_state_dict['model'][prefix+'out_weight.weight']
                #state_dict[new_name + "out_weight.weight"] =  krail_state_dict['model']['encoder.layers.23.fc2.weight']
            if prefix+'out_weight.bias' not in state_dict:
                # krail_tensor = krail_state_dict['model']['decoder.layers.'+str(layer_num*split_num)+'.fc2.bias']
                # for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                #     krail_tensor += krail_state_dict['model']['decoder.layers.'+str(i)+'.fc2.bias']
                # krail_tensor = krail_tensor / split_num
                # state_dict[prefix + "out_weight.bias"] =  krail_tensor
                state_dict[prefix + "out_weight.bias"] =  krail_state_dict['model'][prefix+'out_weight.bias']
                # #state_dict[new_name + "out_weight.bias"] =  krail_state_dict['model']['encoder.layers.23.fc2.bias']
            if prefix+'adapter_layer_norm.weight' not in state_dict:
                # krail_tensor = krail_state_dict['model']['decoder.layers.'+str(layer_num*split_num)+'.final_layer_norm.weight']
                # for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                #     krail_tensor += krail_state_dict['model']['decoder.layers.'+str(i)+'.final_layer_norm.weight']
                # krail_tensor = krail_tensor / split_num
                # state_dict[prefix + "adapter_layer_norm.weight"] =  krail_tensor
                state_dict[prefix + "adapter_layer_norm.weight"] =  krail_state_dict['model'][prefix+'adapter_layer_norm.weight']
                #state_dict[new_name + "adapter_layer_norm.weight"] =  krail_state_dict['model']['encoder.layers.23.final_layer_norm.weight']
            if prefix+'adapter_layer_norm.bias' not in state_dict:
                # krail_tensor = krail_state_dict['model']['decoder.layers.'+str(layer_num*split_num)+'.final_layer_norm.bias']
                # for i in range(layer_num*split_num+1,layer_num*split_num+split_num):
                #     krail_tensor += krail_state_dict['model']['decoder.layers.'+str(i)+'.final_layer_norm.bias']
                # krail_tensor = krail_tensor / split_num
                # state_dict[prefix + "adapter_layer_norm.bias"] =  krail_tensor
                state_dict[prefix + "adapter_layer_norm.bias"] =  krail_state_dict['model'][prefix+'adapter_layer_norm.bias']
                # #state_dict[new_name + "adapter_layer_norm.bias"] =  krail_state_dict['model']['encoder.layers.23.final_layer_norm.bias']
        elif self.cfg.adapter_module != 'none' and krail_state_dict is not None and self.cfg.krail_module == 'krail_adaptive_final':
            layer_num = int(prefix.split('.')[-2])
            print('----- load krail process done! Overwrite init adapters of '+prefix+' with a map strategy of PLM_adapter ! -----')
            split_num = 2
            if prefix+'in_weight.weight' not in state_dict:
                state_dict[prefix + "in_weight.weight"] =  krail_state_dict['model']['decoder.layers.23.in_weight.weight']         
            if prefix+'in_weight.bias' not in state_dict:
                state_dict[prefix + "in_weight.bias"] =  krail_state_dict['model']['decoder.layers.23.in_weight.bias']
            if prefix+'out_weight.weight' not in state_dict:
                state_dict[prefix + "out_weight.weight"] =  krail_state_dict['model']['decoder.layers.23.out_weight.weight']
            if prefix+'out_weight.bias' not in state_dict:
                state_dict[prefix + "out_weight.bias"] =  krail_state_dict['model']['decoder.layers.23.out_weight.bias']
            if prefix+'adapter_layer_norm.weight' not in state_dict:
                state_dict[prefix + "adapter_layer_norm.weight"] =  krail_state_dict['model']['decoder.layers.23.adapter_layer_norm.weight']
            if prefix+'adapter_layer_norm.bias' not in state_dict:
                state_dict[prefix + "adapter_layer_norm.bias"] =  krail_state_dict['model']['decoder.layers.23.adapter_layer_norm.bias']
        elif self.cfg.adapter_module != 'none' and krail_state_dict is None:
            if prefix+'in_weight.weight' not in state_dict:
                state_dict[prefix + "in_weight.weight"] =  self.in_weight.weight
            if prefix+'in_weight.bias' not in state_dict:
                state_dict[prefix + "in_weight.bias"] =  self.in_weight.bias
            if prefix+'out_weight.weight' not in state_dict:
                state_dict[prefix + "out_weight.weight"] =  self.out_weight.weight
            if prefix+'out_weight.bias' not in state_dict:
                state_dict[prefix + "out_weight.bias"] =  self.out_weight.bias
            if prefix+'adapter_layer_norm.weight' not in state_dict:
                state_dict[prefix + "adapter_layer_norm.weight"] =  self.adapter_layer_norm.weight
            if prefix+'adapter_layer_norm.bias' not in state_dict:
                state_dict[prefix + "adapter_layer_norm.bias"] =  self.adapter_layer_norm.bias
        
        super(TransformerDecoderLayerAdapter, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    
    
    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, cfg, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            dropout=cfg.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not cfg.cross_self_attention,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=cfg.decoder.xformers_att_config,
        )

    def build_encoder_attention(self, embed_dim, cfg):
        return MultiheadAttention(
            embed_dim,
            cfg.decoder.attention_heads,
            kdim=cfg.encoder.embed_dim,
            vdim=cfg.encoder.embed_dim,
            dropout=cfg.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            xformers_att_config=cfg.encoder.xformers_att_config,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            #print('encoder out for cross attn --------------------------:')
            
            #print(encoder_out)

            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        if self.c_attn is not None:
            tgt_len, bsz = x.size(0), x.size(1)
            x = x.view(tgt_len, bsz, self.nh, self.head_dim)
            x = torch.einsum("tbhd,h->tbhd", x, self.c_attn)
            x = x.reshape(tgt_len, bsz, self.embed_dim)
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
        
        residual = x
        # if self.normalize_before:
        #     x = self.final_layer_norm(x)
        if self.cfg.adapter_module == 'parallel_norm':
            if self.normalize_before:
                x = self.final_layer_norm(x)
                px = self.adapter_layer_norm(x)
        else:
            if self.normalize_before:
                x = self.final_layer_norm(x)
    
        # parallel adapter
        if self.cfg.adapter_module == 'parallel' or self.cfg.adapter_module == 'parallel_dp' or self.cfg.adapter_module == 'parallel_norm':
            # residual = px
            # if self.normalize_before:
            #     px = self.adapter_layer_norm(px)
            if self.cfg.adapter_module == 'parallel_norm':
                px = self.in_weight(px)
            else:
                px = self.in_weight(x)

            if self.cfg.no_adapter_relu:
                pass
            else:
                px = self.adapter_relu(px)
            px = self.out_weight(px)
            if self.cfg.adapter_module == 'parallel_dp':
                pass
            else:
                px = self.dropout_module(px)
            # px = self.residual_connection(px, residual)
            # if not self.normalize_before:
                # px = self.final_layer_norm(px)

            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            if self.ffn_layernorm is not None:
                x = self.ffn_layernorm(x)
            x = self.fc2(x)
            x = x + px
        elif self.cfg.adapter_module == 'parallel_gate':
                px = self.in_weight(x)
                if self.cfg.no_adapter_relu:
                    pass
                else:
                    px = self.adapter_relu(px)
                px = self.out_weight(px)
                px = self.dropout_module(px)

                gx = torch.cat([x, px], dim=-1)

                gx = self.sigmod_gate(self.attn_gate(gx))

                x = x * gx + px * (1-gx)
        else:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            if self.ffn_layernorm is not None:
                x = self.ffn_layernorm(x)
            x = self.fc2(x)
        
        x = self.dropout_module(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        # serial adapter
        if self.cfg.adapter_module == 'serial' or self.cfg.adapter_module == 'serial_single':
            residual = x
            if self.normalize_before:
                x = self.adapter_layer_norm(x)
            x = self.in_weight(x)
            if self.cfg.no_adapter_relu:
                pass
            else:
                x = self.adapter_relu(x)
            x = self.out_weight(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.final_layer_norm(x)


        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        
        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


# backward compatible with the legacy argparse format
class TransformerDecoderLayer(TransformerDecoderLayerAdapter):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(
            TransformerConfig.from_namespace(args),
            no_encoder_attn=no_encoder_attn,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.args = args

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return super().build_self_attention(
            embed_dim,
            TransformerConfig.from_namespace(args),
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

    def build_encoder_attention(self, embed_dim, args):
        return super().build_encoder_attention(
            embed_dim,
            TransformerConfig.from_namespace(args),
        )
