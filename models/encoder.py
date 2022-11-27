# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math

def build_encoder_FoG(d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate=False):
    
    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    return TransformerEncoder_with_intermid(encoder_layer, num_encoder_layers, norm=encoder_norm, 
                return_intermidiate=return_intermediate)

class TransformerEncoder_with_intermid(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermidiate=True):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.d_model = self.layers[0].d_model
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermidiate = return_intermidiate
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # breakpoint()
        src = src.permute(2, 0, 1) # [L, bs, d_model] for batch_first default to False
        pos = pos.permute(2, 0, 1) if pos is not None else None
        
        output = src

        intermidiate, attn_weights = [], []
        for layer in self.layers:
            if self.return_intermidiate:
                intermidiate.append(self.norm(output) if self.norm is not None else output)
            output, attn_weight_now = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            attn_weights.append(attn_weight_now)  # (N,num_heads,L,S)
            
        intermidiate.append(self.norm(output) if self.norm is not None else output)
        intermidiate = torch.stack(intermidiate, dim=0)
        attn_weights = torch.stack(attn_weights, dim=0)  # (num_layers, N,num_heads,L,S)
        if self.return_intermidiate:
            return intermidiate.transpose(1, 2), attn_weights

        return intermidiate[-1].unsqueeze(0).transpose(1, 2), attn_weights

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):

        bs, c, L = src.shape
        src = src.permute(2, 0, 1)
        pos_embed = pos_embed.permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        tgt = torch.zeros_like(query_embed)
        # print(mask.device, tgt.device)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, L)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.d_model = d_model
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = MultiHeadAttention_with_constant(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     average_attn_weights = True):
        q = k = self.with_pos_embed(src, pos)
        src2, attn_weights = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask,
                average_attn_weights=average_attn_weights)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights # (N,num_heads,L,S)

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    average_attn_weights = True):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, attn_weights = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask, need_weights=True,
                              average_attn_weights=average_attn_weights)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, attn_weights # (N,num_heads,L,S)

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                average_attn_weights = False):
        # breakpoint()
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos, average_attn_weights)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos, average_attn_weights)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer():
    hidden_dim, dropout, nheads = 256, 0.1, 8
    dim_feedforward, enc_layers, dec_layers = 2*hidden_dim, 6, 6
    pre_norm = False
    return Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        normalize_before=pre_norm,
        return_intermediate_dec=True,
    )

class MultiHeadAttention_with_constant(nn.Module):
    def __init__(self, d_model, nhead, seq_len=32, constant_head=0, dropout=0.1, batch_first=False) -> None:
        super().__init__()
        assert d_model % nhead == 0
        assert nhead >= constant_head
        self.d_model = d_model
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.constant_head = constant_head
        self.seq_len = seq_len
        self.batch_first = batch_first # False: q, k, v with dim [L, N, d_model]; True: [N, L, d_model]
        self.dropout = nn.Dropout(p=dropout)
        self.linears = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for _ in range(4)])
        self.constant_scores = nn.Parameter(torch.rand(self.constant_head, self.seq_len, self.seq_len)) if 0 < self.constant_head <= self.nhead else None

    def forward(self, query, key, value, **kwargs):
        # breakpoint()
        if self.batch_first:
            bs, seq_len, d_model = value.shape
        else:
            seq_len, bs, d_model = value.shape
            query, key, value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)
        if self.seq_len != seq_len and self.constant_scores is not None:
            print("Note: Predefined seq_len not equal to given data, replace seq_len {} with {}.".format(self.seq_len, seq_len))
            self.seq_len = seq_len
            self.constant_scores = nn.Parameter(torch.rand(self.constant_head, self.seq_len, self.seq_len).to(value.device))
        
        q_nhead, k_nhead, v_nhead = [l(x) for l, x in zip(self.linears, (query, key, value))]
        q_nhead, k_nhead, v_nhead = [x.view(bs, -1, self.nhead, self.d_k).transpose(1, 2) for x in (q_nhead, k_nhead, v_nhead)]

        if self.constant_head < self.nhead:
            scores = torch.matmul(q_nhead[:, self.constant_head:self.nhead], k_nhead[:, self.constant_head:self.nhead].transpose(-2, -1)) / math.sqrt(self.d_k)
        elif self.constant_head == self.nhead:
            scores = None
        else:
            raise ValueError("Error! constant head bigger than nhead.")

        if self.constant_scores is not None:
            constant_scores = self.constant_scores.repeat((bs, 1, 1, 1))
            scores = torch.cat((constant_scores, scores), dim=1) if scores is not None else constant_scores
        scores = F.softmax(scores, dim=-1)
        scores = self.dropout(scores)
        results = torch.matmul(scores, v_nhead)
        results = results.transpose(1, 2).contiguous().view(bs, -1, self.nhead*self.d_k) # concatenate all heads
        results = self.linears[-1](results)
        return results if self.batch_first else results.transpose(0, 1), scores

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = build_transformer().to(device)
    # print(model)
    
    src = torch.rand(4, 256, 17).to(device) # [L, bs, d_model]
    mask = torch.full((4, 17), False).to(device)
    query_embed = nn.Embedding(17, 256).to(device)
    pos_embed = torch.rand(4, 256, 17).to(device)
    model = build_encoder_FoG(d_model=src.shape[1], dim_feedforward=4*src.shape[1], normalize_before=True, return_intermediate=True).to(device)

    breakpoint()
    # y = model(src, mask, query_embed.weight, pos_embed)
    y = model(src, src_key_padding_mask=None, pos=pos_embed)
    print(y[0].shape, y[1].shape)