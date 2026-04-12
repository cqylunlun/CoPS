from lib.simple_tokenizer import SimpleTokenizer as _Tokenizer
from lib.clip import LayerNorm, QuickGELU
from collections import OrderedDict
from typing import Union, List

import torch
import torch.nn as nn

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if torch.__version__ < '1.8.0':
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('LayerNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.fill_(0)


class Projector(nn.Module):
    def __init__(self, dim_in, dim_out, features_list):
        super(Projector, self).__init__()
        self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for _ in features_list])
        self.apply(weights_init)

    def forward(self, tokens):
        tokens_list = []
        for i in range(len(tokens)):
            tokens_list.append(self.fc[i](tokens[i]))
        return tokens_list


class Prototype_Extractor(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super(Prototype_Extractor, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_1 = LayerNorm(d_model)
        self.ln_2 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model, d_model))
        ]))
        self.apply(weights_init)

    def attention(self, prt, image):
        return self.attn(prt, image, image, need_weights=False)[0]

    def forward(self, x, image):
        x = x.unsqueeze(0).repeat((image.shape[0], 1, 1))
        x = x + self.attention(self.ln_1(x), self.ln_1(image))
        x = x + self.mlp(self.ln_2(x))
        return x


class VAE_Encoder(nn.Module):
    def __init__(self, ctx_dim, dim=768):
        super(VAE_Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, dim),  # [1, 4]
            nn.ReLU(),
        )
        self.mean = nn.Linear(dim, ctx_dim)  # [4, 1]
        self.log_var = nn.Linear(dim, ctx_dim)  # [4, 1]
        self.apply(weights_init)

    def forward(self, x):
        x = self.net(x)  # [8, 768]
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var


class VAE_Decoder(nn.Module):
    def __init__(self, ctx_dim, dim=768):
        super(VAE_Decoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(ctx_dim, dim),  # [1, 8]
            nn.ReLU(),
            nn.Linear(dim, ctx_dim),  # [8, 1]
        )
        self.apply(weights_init)

    def forward(self, x):
        out = self.net(x)  # [8, 768]
        return out


class PromptLearner(nn.Module):
    def __init__(self, clip_model, design_details, out_layers):
        super().__init__()
        self.vae_len = design_details["vae_length"]
        self.cls_len = 2 if self.vae_len != 0 else 0
        self.wo_cls_len = design_details["prompt_length"]
        self.prt_length = design_details["prt_length"]
        self.n_ctx = self.wo_cls_len + self.cls_len
        self.text_encoder_n_ctx = design_details["learnable_text_embedding_length"]
        self.length = clip_model.context_length
        self.dtype = clip_model.transformer.get_cast_dtype()
        ctx_dim = clip_model.ln_final.weight.shape[0]

        print("Initializing learnable prompts")
        self.vae_len = 1 if self.vae_len == 0 else self.vae_len
        self.ctx_pos = nn.Parameter(torch.empty(1, self.vae_len, self.n_ctx, ctx_dim, dtype=self.dtype))  # [1, 1, 12, 768]
        self.ctx_neg = nn.Parameter(torch.empty(1, self.vae_len, self.n_ctx, ctx_dim, dtype=self.dtype))  # [1, 1, 12, 768]
        nn.init.normal_(self.ctx_pos, std=0.02)
        nn.init.normal_(self.ctx_neg, std=0.02)

        prompt_prefix_pos = " ".join(["X"] * self.n_ctx)
        prompt_prefix_neg = " ".join(["X"] * self.n_ctx)
        prompts_pos = [prompt_prefix_pos + " object."] if self.cls_len == 0 else [prompt_prefix_pos + '.']
        prompts_neg = [prompt_prefix_neg + " object."] if self.cls_len == 0 else [prompt_prefix_neg + '.']

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []
        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(tokenize(p_pos))
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(tokenize(p_neg))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)  # [1, 77]
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)  # [1, 77]

        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(self.dtype)  # [1, 77, 768]
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(self.dtype)  # [1, 77, 768]
            n, l, d = embedding_pos.shape  # n=1 l=77 d=768
            embedding_pos = embedding_pos.reshape(1, 1, l, d).permute(1, 0, 2, 3)  # [1, 1, 77, 768]
            embedding_neg = embedding_neg.reshape(1, 1, l, d).permute(1, 0, 2, 3)  # [1, 1, 77, 768]

        # trainable MLP
        self.projector = Projector(1024, 768, out_layers)  # patch-level projection
        self.projector_cls = Projector(768, 768, [24])  # image-level projection

        # Prototype Extractor
        self.extractor = Prototype_Extractor(ctx_dim, 12)
        self.extractor_ = Prototype_Extractor(ctx_dim, 12)
        self.t_n = nn.Parameter(torch.empty(self.prt_length, ctx_dim))  # [6, 768]
        self.t_a = nn.Parameter(torch.empty(self.prt_length, ctx_dim))  # [6, 768]
        nn.init.normal_(self.t_n, std=0.02)
        nn.init.normal_(self.t_a, std=0.02)

        # VAE Encoder and Decoder
        self.vae_encoder = VAE_Encoder(ctx_dim, ctx_dim)
        self.vae_decoder = VAE_Decoder(ctx_dim, ctx_dim)

        n, d = tokenized_prompts_pos.shape  # n=1 d=77
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(1, 1, d).permute(1, 0, 2)  # [1, 1, 77]
        n, d = tokenized_prompts_neg.shape  # n=1 d=77
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(1, 1, d).permute(1, 0, 2)  # [1, 1, 77]

        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)  # [1, 1, 77]
        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :])  # [1, 1, 1, 768]
        self.register_buffer("token_suffix_pos", embedding_pos[:, :, 1 + self.n_ctx:, :])  # [1, 1, 64, 768]
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)  # [1, 1, 77]
        self.register_buffer("token_prefix_neg", embedding_neg[:, :, :1, :])  # [1, 1, 1, 768]
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + self.n_ctx:, :])  # [1, 1, 64, 768]

        # learnable token in Text Encoder
        if self.text_encoder_n_ctx != 0:
            self.compound_prompts_depth = design_details["learnable_text_embedding_depth"]
            self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(self.text_encoder_n_ctx, ctx_dim))
                                                           for _ in range(self.compound_prompts_depth)])
            for single_para in self.compound_prompts_text:
                nn.init.normal_(single_para, std=0.02)
        else:
            self.compound_prompts_text = None

    def forward(self, clip_model, batchnum, agg_prototype, bias, cls_name):
        ctx_pos = self.ctx_pos.repeat(batchnum, 1, 1, 1)  # [8, 1, 12, 768]
        ctx_neg = self.ctx_neg.repeat(batchnum, 1, 1, 1)  # [8, 1, 12, 768]
        prefix_pos = self.token_prefix_pos.repeat(1, self.vae_len, 1, 1)  # [1, 1, 1, 768]
        prefix_neg = self.token_prefix_neg.repeat(1, self.vae_len, 1, 1)  # [1, 1, 1, 768]
        suffix_pos = self.token_suffix_pos.repeat(1, self.vae_len, 1, 1)  # [1, 1, 64, 768]
        suffix_neg = self.token_suffix_neg.repeat(1, self.vae_len, 1, 1)  # [1, 1, 64, 768]

        pos_context = ctx_pos[:, :, :self.wo_cls_len - self.prt_length]  # [8, 1, 6, 768]
        neg_context = ctx_neg[:, :, :self.wo_cls_len - self.prt_length]  # [8, 1, 6, 768]
        pos_state = ctx_pos[:, :, self.wo_cls_len - self.prt_length:self.wo_cls_len] + agg_prototype[:, 1:]  # [8, 1, 6, 768]
        neg_state = ctx_neg[:, :, self.wo_cls_len - self.prt_length:self.wo_cls_len] + agg_prototype[:, :1]  # [8, 1, 6, 768]

        compound_prompts_pos, compound_prompts_neg = [], []
        bias = bias.reshape(batchnum, 1, -1, bias.shape[-1])  # [8, 1, 6, 768]
        for i in range(bias.shape[-2]):
            pos_bias = bias[:, :, i, :].unsqueeze(2)  # [8, 1, 1, 768]
            neg_bias = bias[:, :, i, :].unsqueeze(2)  # [8, 1, 1, 768]
            pos_class = ctx_pos[:, :, self.wo_cls_len:] + pos_bias
            neg_class = ctx_neg[:, :, self.wo_cls_len:] + neg_bias
            ctx_pos_ = torch.cat([pos_context, pos_state, pos_class], dim=2)
            ctx_neg_ = torch.cat([neg_context, neg_state, neg_class], dim=2)

            prompts_pos_list = []
            prompts_neg_list = []
            for j in range(batchnum):
                prompts_pos = torch.cat([
                    prefix_pos,  # [1, 1, 1, 768]
                    ctx_pos_[j].unsqueeze(0),  # [1, 1, 12, 768]
                    suffix_pos,  # [1, 1, 64, 768]
                ], dim=2)
                prompts_pos_list.append(prompts_pos)

                prompts_neg = torch.cat([
                    prefix_neg,  # [1, 1, 1, 768]
                    ctx_neg_[j].unsqueeze(0),  # [1, 1, 12, 768]
                    suffix_neg,  # [1, 1, 64, 768]
                ], dim=2)
                prompts_neg_list.append(prompts_neg)
            prompts_pos = torch.cat(prompts_pos_list, dim=0)  # [8, 1, 77, 768]
            prompts_neg = torch.cat(prompts_neg_list, dim=0)  # [8, 1, 77, 768]

            prompts_pos = prompts_pos.reshape(-1, *prompts_pos.shape[-2:])  # [8, 77, 768]
            prompts_neg = prompts_neg.reshape(-1, *prompts_neg.shape[-2:])  # [8, 77, 768]
            compound_prompts_pos.append(prompts_pos)
            compound_prompts_neg.append(prompts_neg)
        compound_prompts_pos = torch.stack(compound_prompts_pos, dim=1)
        compound_prompts_neg = torch.stack(compound_prompts_neg, dim=1)
        compound_prompts_pos = compound_prompts_pos.reshape(batchnum, -1, *compound_prompts_pos.shape[-2:])
        compound_prompts_neg = compound_prompts_neg.reshape(batchnum, -1, *compound_prompts_neg.shape[-2:])
        compound_prompts = torch.cat((compound_prompts_pos, compound_prompts_neg), dim=1)  # [8, 12, 77, 768]

        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1, self.tokenized_prompts_pos.shape[-1])  # [1, 77]
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1, self.tokenized_prompts_neg.shape[-1])  # [1, 77]
        tokenized_prompts_pos = tokenized_prompts_pos.unsqueeze(0).repeat(batchnum * bias.shape[-2] * self.vae_len, 1, 1)  # [8*6, 1, 77]
        tokenized_prompts_neg = tokenized_prompts_neg.unsqueeze(0).repeat(batchnum * bias.shape[-2] * self.vae_len, 1, 1)  # [8*6, 1, 77]
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim=0)  # [8*12, 77]

        text_features = clip_model.encode_text_learn(compound_prompts, tokenized_prompts, self.compound_prompts_text).float()  # [8, 2, 768]
        return text_features
