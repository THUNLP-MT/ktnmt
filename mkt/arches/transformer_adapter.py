# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging.handlers import DEFAULT_UDP_LOGGING_PORT
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    gen_parser_from_dataclass,
)
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models.transformer import (
    TransformerConfig,
)
from .transformer_encoder_adapter import TransformerEncoderBaseAdapter
from .transformer_decoder_adapter import TransformerDecoderBaseAdapter
from ..misc.load_krail import get_add_embed_tensor, load_checkpoint_to_cpu

import logging
logger = logging.getLogger(__name__)
from omegaconf import DictConfig
from argparse import Namespace

krail_map_list = ['krail', 'krail_map', 'krail_map_avg', 'krail_map_last', 'krail_adaptive', 'krail_adaptive_final']

class TransformerModelAdapter(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.encoder = encoder
        self.supports_align_args = True
        
        self.add_vocab_size = 0
        

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens, base_encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            base_decoder_embed_tokens = base_encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens, base_encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens, base_decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )

        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens, base_encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens, base_decoder_embed_tokens)
        return cls(cfg, encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)    
        
        # custom 

        if cfg.base_embed_dim > 0:
            base_embed_dim = cfg.base_embed_dim 
        
            base_emb = Embedding(base_embed_dim, embed_dim, padding_idx)
        else:
            base_emb = None
        
        
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        
        if cfg.freeze_all and base_emb is not None:
            base_emb.weight.requires_grad=False
        elif cfg.freeze_all:
            emb.weight.requires_grad=False
            
        return emb, base_emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens, base_embed_tokens):

        return TransformerEncoderBaseAdapter(cfg, src_dict, embed_tokens, base_embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens, base_embed_tokens):
        return TransformerDecoderBaseAdapter(
            cfg,
            tgt_dict,
            embed_tokens, 
            base_embed_tokens, 
            no_encoder_attn=cfg.no_cross_attention,
        )

    # custom
    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None,
    ):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """

        if model_cfg is None and args is not None:
            logger.warn(
                "using 'args' is deprecated, please update your code to use dataclass config"
            )
            model_cfg = convert_namespace_to_omegaconf(args).model
        
        # custom
        if model_cfg.vocab_adapter and model_cfg.freeze_all and model_cfg.base_embed_dim > 0:
            if 'encoder.base_embed_tokens.weight' in state_dict:
                if state_dict['encoder.base_embed_tokens.weight'].size(0) == state_dict['encoder.embed_tokens.weight'].size(0):
                    state_dict['encoder.base_embed_tokens.weight'] = state_dict['encoder.embed_tokens.weight']
                    state_dict['decoder.base_embed_tokens.weight'] = state_dict['decoder.embed_tokens.weight']
            else:
                state_dict['encoder.base_embed_tokens.weight'] = state_dict['encoder.embed_tokens.weight']
                state_dict['decoder.base_embed_tokens.weight'] = state_dict['decoder.embed_tokens.weight']
        if model_cfg.krail_model:
            if model_cfg.krail_module in krail_map_list or model_cfg.krail_module == 'krail_embed':
                krail_state_dict = load_checkpoint_to_cpu(model_cfg.krail_model)
            else:
                krail_state_dict = None
        else:
            krail_state_dict = None
        if model_cfg.vocab_adapter:
            torch.manual_seed(model_cfg.seed)
            
            load_state_num_embeddings = state_dict['encoder.embed_tokens.weight'].size(0)
            self.add_vocab_size = self.encoder.embed_tokens.num_embeddings-load_state_num_embeddings
            
            if self.add_vocab_size > 0:
                addition_embeds_tensor = torch.zeros(self.add_vocab_size, model_cfg.encoder_embed_dim)
                if model_cfg.krail_module in krail_map_list or model_cfg.krail_module == 'krail_embed':
                    
                    addition_embeds_tensor = get_add_embed_tensor(model_cfg.new_dict_path, model_cfg.krail_dict_path, addition_embeds_tensor, krail_state_dict['model']['encoder.embed_tokens.weight'], load_state_num_embeddings)
                    print(addition_embeds_tensor)
                else:
                    nn.init.normal_(addition_embeds_tensor, mean=0, std=5 ** -0.5)

                state_dict['encoder.embed_tokens.weight'] = torch.cat((state_dict['encoder.embed_tokens.weight'], addition_embeds_tensor), dim=0)
                state_dict['decoder.embed_tokens.weight'] = torch.cat((state_dict['decoder.embed_tokens.weight'], addition_embeds_tensor), dim=0)
                state_dict['decoder.output_projection.weight'] = torch.cat((state_dict['decoder.output_projection.weight'], addition_embeds_tensor), dim=0)
                self.cfg.add_embed_dim = self.add_vocab_size
                

            else:
                pass
            
        else:
            pass
        
        
        self.upgrade_state_dict(state_dict)

        from fairseq.checkpoint_utils import prune_state_dict

        new_state_dict = prune_state_dict(state_dict, model_cfg)
        
        return super().load_state_dict(new_state_dict, strict)


    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m
