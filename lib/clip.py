from collections import OrderedDict
from typing import Tuple, Union
from torch import nn

import torch
import numpy as np


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# q-kv attention and v-vv attention
class Attention(nn.Module):
    def __init__(self, out_dim, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., settings=''):
        super().__init__()
        self.num_heads = num_heads  # 16
        head_dim = dim // num_heads  # 64
        self.scale = qk_scale or head_dim ** -0.5  # 0.125

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.settings = settings

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # qk-v self-attention for the original ViT
        attn_ori = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)
        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x_ori = self.proj_drop(self.proj(x_ori))

        # replace k & q by v, perform v-v attention
        q = k = v
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))

        return [x, x_ori]


# frozen block in Vision Encoder
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if isinstance(self.attn, Attention):
            x = x.transpose(0, 1)
            x, x_ori = self.attn(x)
            return [x.transpose(0, 1), x_ori.transpose(0, 1)]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x, ffn=False):
        # dual paths for blocks deeper than "DPAM_layer"
        if isinstance(self.attn, Attention):
            if isinstance(x, list):
                if not ffn:  # skip ffn for the new path
                    x, x_ori = x
                    x_res = self.attention(self.ln_1(x_ori))
                    x_res, x_ori_res = x_res
                    x_ori += x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res
                    return [x, x_ori]
                else:
                    x, x_ori_1 = x
                    x_res = self.attention(self.ln_1(x_ori_1))
                    x_res, x_ori_res = x_res
                    x_ori = x_ori_1 + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x = x_res + x_ori_1
                    x = x + self.mlp(self.ln_2(x))
                    return [x, x_ori]

            # start of dual path
            else:
                x_res = self.attention(self.ln_1(x))  # x -> [1370, 8, 1024]
                x_res, x_ori_res = x_res
                x_ori = x + x_ori_res
                x_ori = x_ori + self.mlp(self.ln_2(x_ori))  # ori block output
                x += x_res  # new block output
                return [x, x_ori]

        # single path before "DPAM_layer"
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x


# learnable block in Text Encoder
class ResidualAttentionBlock_learnable_token(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None,
                 text_layer=False, i=0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.i = i
        self.compound_prompt_nctx = design_details['learnable_text_embedding_length']
        self.text_layer = text_layer
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs):
        x = inputs[0]  # [77, 2, 768]
        compound_prompts_deeper = inputs[1]
        counter = inputs[2]

        if not (self.first_layer or counter > len(compound_prompts_deeper) - 1):
            prefix = x[:1, :, :]
            suffix = x[1 + self.compound_prompt_nctx:, :, :]
            textual_context = compound_prompts_deeper[counter]
            textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
            x = torch.cat([prefix, textual_context, suffix], dim=0)
            counter += 1

        x = x + self.attention(self.ln_1(x))  # [77, 2, 768]
        x = x + self.mlp(self.ln_2(x))  # [77, 2, 768]
        return [x, compound_prompts_deeper, counter]


# Vanilla Transformer
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, design_details=None, text_layer=False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.text_layer = text_layer
        self.design_details = design_details
        if self.text_layer and self.design_details:
            self.resblocks = nn.ModuleList([ResidualAttentionBlock_learnable_token(width, heads, attn_mask,
                                                                                   design_details, text_layer, i=i) for i in range(layers)])
        else:
            self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask, ) for i in range(layers)])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def ori_CLIP_with_patch_forward(self, x, out_layers):
        idx = 0
        global x_
        out_tokens = []
        for r in self.resblocks:
            idx += 1
            x = r(x)
            if idx in out_layers:
                if isinstance(x, list):
                    out_tokens.append(x[1].clone())  # x_ori (original block)
                else:
                    out_tokens.append(x.clone())
        return x, out_tokens

    def CoPS_forward(self, x, out_layers, ffn):
        idx = 0
        global x_
        out_tokens = []
        for r in self.resblocks:
            idx += 1
            x = r(x, ffn=ffn)
            if idx in out_layers:
                if isinstance(x, list):
                    out_tokens.append(x[0].clone())  # x (new block)
                else:
                    out_tokens.append(x.clone())
        return x, out_tokens

    def forward(self, x: torch.Tensor, out_layers=[24], DPAM_layer=None, ffn=False):
        # frozen vision encoder forward
        if not self.text_layer:  # frozen ViT
            if DPAM_layer is None:  # q-kv attention
                x, out_tokens = self.ori_CLIP_with_patch_forward(x, out_layers)
            else:  # v-vv attention
                x, out_tokens = self.CoPS_forward(x, out_layers, ffn)
            return x, out_tokens

        # ori text encoder forward
        elif self.design_details is None:
            for idx, r in enumerate(self.resblocks):
                x = r(x)
            return x

        # insert learnable text encoder forward
        elif self.design_details is not None:
            for idx, r in enumerate(self.resblocks):
                x = r(x)
            return x[0]  # [77, 2, 768]
        return None


# Vision Encoder
class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)
        self.attn = None
        self.embed_dim = width
        self.num_heads = heads

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    # original q-kv weight -> DPAM v-vv weight
    @torch.no_grad()
    def DPAM_replace(self, DPAM_layer):
        if DPAM_layer is not None:
            for i in range(1, DPAM_layer + 1):
                self.attn = Attention(self.embed_dim, self.embed_dim, self.num_heads, True)
                self.attn.qkv.weight.data = self.transformer.resblocks[-i].attn.in_proj_weight.clone()
                self.attn.qkv.bias.data = self.transformer.resblocks[-i].attn.in_proj_bias.clone()
                self.attn.proj.weight.data = self.transformer.resblocks[-i].attn.out_proj.weight.clone()
                self.attn.proj.bias.data = self.transformer.resblocks[-i].attn.out_proj.bias.clone()
                self.transformer.resblocks[-i].attn = self.attn

    @torch.no_grad()
    def forward(self, x: torch.Tensor, features_list, DPAM_layer=None, ffn=False):
        x = self.conv1(x)  # [8, 3, 518, 518] -> [8, 1024, 37, 37]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [8, 1024, 37, 37] -> [8, 1024, 1369]
        x = x.permute(0, 2, 1)  # [8, 1024, 1369] -> [8, 1369, 1024]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros_like(x[:, :1]), x], dim=1)  # [8, 1369, 1024] -> [8, 1370, 1024]

        # update the position embedding during inference for varied input size 24->37 / 336->518
        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)  # 24
        new_side = int((x.shape[1] - 1) ** 0.5)  # 37
        if side != new_side:
            new_pos = self.positional_embedding[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode='bilinear')
            new_pos = new_pos.reshape(-1, x.shape[-1], new_side * new_side).transpose(1, 2)
            self.positional_embedding.data = torch.cat([self.positional_embedding[:1, :], new_pos[0]], 0)

        pos = self.positional_embedding.to(x.dtype)  # [1370, 1024]
        x = x + pos  # [8, 1370, 1024]
        x = self.ln_pre(x)  # [8, 1370, 1024]
        x = x.permute(1, 0, 2)  # [1370, 8, 1024]

        if DPAM_layer != None:
            [x, x_ori], patch_tokens = self.transformer(x, features_list, DPAM_layer=DPAM_layer, ffn=ffn)
        else:
            x_ori, patch_tokens = self.transformer(x, features_list, DPAM_layer=DPAM_layer)

        image_tokens = x_ori[0] @ self.proj
        patch_tokens = [self.ln_post(patch_token.permute(1, 0, 2))[:, 1:] @ self.proj for patch_token in patch_tokens]
        return image_tokens, patch_tokens, image_tokens, patch_tokens


class CoPS(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details=None
                 ):
        super().__init__()

        self.context_length = context_length  # 77

        # frozen vision encoder
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        # learnable text encoder
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            text_layer=True,
            design_details=design_details
        )

        self.vocab_size = vocab_size  # 49408
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    # lazily create causal attention mask, with full attention between the vision tokens
    # pytorch uses additive attention mask; fill with -inf
    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    # image encoder
    def encode_image(self, image, feature_list, DPAM_layer=None, ffn=False):
        return self.visual(image.type(self.dtype), feature_list, DPAM_layer=DPAM_layer, ffn=ffn)

    # text encoder with +/- input
    def encode_text(self, prompts, tokenized_prompts, deep_compound_prompts_text=None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        if deep_compound_prompts_text is None:  # ori text encoder
            x = self.transformer(x)
        else:  # insert learnable text encoder
            x = self.transformer([x, deep_compound_prompts_text, 0])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

    # text encoder with +&- input
    def encode_text_learn(self, prompts, tokenized_prompts, deep_compound_prompts_text=None):
        cast_dtype = self.transformer.get_cast_dtype()
        x = prompts + self.positional_embedding.to(cast_dtype)  # [8, 12, 77, 768]
        x = x.reshape(-1, *x.shape[-2:])  # [16, 77, 768]
        x = x.permute(1, 0, 2)  # [77, 16, 768]
        if deep_compound_prompts_text is None:  # ori text encoder
            x = self.transformer(x)
        else:  # insert learnable text encoder
            x = self.transformer([x, deep_compound_prompts_text, 0])
        x = x.permute(1, 0, 2)  # [16, 77, 768]
        x = self.ln_final(x).type(self.dtype)  # [16, 77, 768]

        tokenized_prompts = tokenized_prompts.reshape(-1, tokenized_prompts.shape[-1])  # [16, 77]
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection  # [16, 768]
        x = x / x.norm(dim=-1, keepdim=True)
        x = x.reshape(*prompts.shape[:2], -1)  # [8, 12, 768]
        x = x.reshape(x.shape[0], 2, -1, x.shape[-1])  # [8, 2, 6, 768]
        return x
