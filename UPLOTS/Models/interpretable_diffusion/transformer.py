import math
import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from einops import rearrange, reduce, repeat
from Models.interpretable_diffusion.model_utils import LearnablePositionalEncoding, Conv_MLP,\
                                                       AdaLayerNorm, Transpose, GELU2, series_decomp
from einops import rearrange
from Models.interpretable_diffusion.language_backbone import (
    hidden_states_from_output,
    load_language_backbone,
)

class TrendBlock(nn.Module):
    """
    Model trend of time series using the polynomial regressor.
    """
    def __init__(self, in_dim, out_dim, in_feat, out_feat, act):
        super(TrendBlock, self).__init__()
        trend_poly = 3
        self.trend = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=trend_poly, kernel_size=3, padding=1),
            act,
            Transpose(shape=(1, 2)),
            nn.Conv1d(in_feat, out_feat, 3, stride=1, padding=1)
        )

        lin_space = torch.arange(1, out_dim + 1, 1) / (out_dim + 1)
        self.poly_space = torch.stack([lin_space ** float(p + 1) for p in range(trend_poly)], dim=0)

    def forward(self, input):
        b, c, h = input.shape
        x = self.trend(input).transpose(1, 2)
        trend_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        trend_vals = trend_vals.transpose(1, 2)
        return trend_vals


class MovingBlock(nn.Module):
    """
    Model trend of time series using the moving average.
    """
    def __init__(self, out_dim):
        super(MovingBlock, self).__init__()
        size = max(min(int(out_dim / 4), 24), 4)
        self.decomp = series_decomp(size)

    def forward(self, input):
        b, c, h = input.shape
        x, trend_vals = self.decomp(input)
        return x, trend_vals


class FourierLayer(nn.Module):
    """
    Model seasonality of time series using the inverse DFT.
    """
    def __init__(self, d_model, low_freq=1, factor=1):
        super().__init__()
        self.d_model = d_model
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs(), 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple


class SeasonBlock(nn.Module):
    """
    Model seasonality of time series using the Fourier series.
    """
    def __init__(self, in_dim, out_dim, factor=1):
        super(SeasonBlock, self).__init__()
        season_poly = factor * min(32, int(out_dim // 2))
        self.season = nn.Conv1d(in_channels=in_dim, out_channels=season_poly, kernel_size=1, padding=0)
        fourier_space = torch.arange(0, out_dim, 1) / out_dim
        p1, p2 = (season_poly // 2, season_poly // 2) if season_poly % 2 == 0 \
            else (season_poly // 2, season_poly // 2 + 1)
        s1 = torch.stack([torch.cos(2 * np.pi * p * fourier_space) for p in range(1, p1 + 1)], dim=0)
        s2 = torch.stack([torch.sin(2 * np.pi * p * fourier_space) for p in range(1, p2 + 1)], dim=0)
        self.poly_space = torch.cat([s1, s2])

    def forward(self, input):
        b, c, h = input.shape
        x = self.season(input)
        season_vals = torch.matmul(x.transpose(1, 2), self.poly_space.to(x.device))
        season_vals = season_vals.transpose(1, 2)
        return season_vals


class FullAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(self,
                 n_embd, # the embed dim
                 condition_embd, # condition dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(condition_embd, n_embd)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class EncoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU'
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
            )

        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.mlp = nn.Sequential(
                nn.Linear(n_embd, mlp_hidden_times * n_embd),
                act,
                nn.Linear(mlp_hidden_times * n_embd, n_embd),
                nn.Dropout(resid_pdrop),
            )

    def forward(self, x, timestep, mask=None, label_emb=None):
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))   # only one really use encoder_output
        return x, att


class Encoder(nn.Module):
    def __init__(
        self,
        n_layer=14,
        n_embd=1024,
        n_head=16,
        attn_pdrop=0.,
        resid_pdrop=0.,
        mlp_hidden_times=4,
        block_activate='GELU',
    ):
        super().__init__()

        self.blocks = nn.Sequential(*[EncoderBlock(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
        ) for _ in range(n_layer)])

    def forward(self, input, t, padding_masks=None, label_emb=None):
        x = input
        for block_idx in range(len(self.blocks)):
            x, _ = self.blocks[block_idx](x, t, mask=padding_masks, label_emb=label_emb)
        return x


class DecoderBlock(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self,
                 n_channel,
                 n_feat,
                 n_embd=1024,
                 n_head=16,
                 attn_pdrop=0.1,
                 resid_pdrop=0.1,
                 mlp_hidden_times=4,
                 activate='GELU',
                 condition_dim=1024,
                 ):
        super().__init__()

        self.ln1 = AdaLayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn1 = FullAttention(
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                )
        self.attn2 = CrossAttention(
                n_embd=n_embd,
                condition_embd=condition_dim,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                )

        self.ln1_1 = AdaLayerNorm(n_embd)

        assert activate in ['GELU', 'GELU2']
        act = nn.GELU() if activate == 'GELU' else GELU2()

        self.trend = TrendBlock(n_channel, n_channel, n_embd, n_feat, act=act)
        # self.decomp = MovingBlock(n_channel)
        self.seasonal = FourierLayer(d_model=n_embd)
        # self.seasonal = SeasonBlock(n_channel, n_channel)

        self.mlp = nn.Sequential(
            nn.Linear(n_embd, mlp_hidden_times * n_embd),
            act,
            nn.Linear(mlp_hidden_times * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

        self.proj = nn.Conv1d(n_channel, n_channel * 2, 1)
        self.linear = nn.Linear(n_embd, n_feat)

    def forward(self, x, encoder_output, timestep, mask=None, label_emb=None):
        a, att = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        a, att = self.attn2(self.ln1_1(x, timestep), encoder_output, mask=mask)
        x = x + a
        x1, x2 = self.proj(x).chunk(2, dim=1)
        trend, season = self.trend(x1), self.seasonal(x2)
        x = x + self.mlp(self.ln2(x))
        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m), trend, season


class Decoder(nn.Module):
    def __init__(
        self,
        n_channel,
        n_feat,
        n_embd=1024,
        n_head=16,
        n_layer=10,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        condition_dim=512
    ):
      super().__init__()
      self.d_model = n_embd
      self.n_feat = n_feat
      self.blocks = nn.Sequential(*[DecoderBlock(
                n_feat=n_feat,
                n_channel=n_channel,
                n_embd=n_embd,
                n_head=n_head,
                attn_pdrop=attn_pdrop,
                resid_pdrop=resid_pdrop,
                mlp_hidden_times=mlp_hidden_times,
                activate=block_activate,
                condition_dim=condition_dim,
        ) for _ in range(n_layer)])

    def forward(self, x, t, enc, padding_masks=None, label_emb=None):
        b, c, _ = x.shape
        # att_weights = []
        mean = []
        season = torch.zeros((b, c, self.d_model), device=x.device)
        trend = torch.zeros((b, c, self.n_feat), device=x.device)
        for block_idx in range(len(self.blocks)):
            x, residual_mean, residual_trend, residual_season = \
                self.blocks[block_idx](x, enc, t, mask=padding_masks, label_emb=label_emb)
            season += residual_season
            trend += residual_trend
            mean.append(residual_mean)

        mean = torch.cat(mean, dim=1)
        return x, mean, trend, season



class Transformer(nn.Module):
    def __init__(
        self,
        n_feat,
        n_channel,
        n_layer_enc=5,
        n_layer_dec=14,
        n_embd=1024,
        n_heads=16,
        attn_pdrop=0.1,
        resid_pdrop=0.1,
        mlp_hidden_times=4,
        block_activate='GELU',
        max_len=2048,
        conv_params=None,
        backbone='gpt2',
        backbone_name=None,
        backbone_layers=None,
        backbone_init='pretrained',
        prompt_mode='text',
        num_prompt_ids=1,
        train_backbone=False,
        **kwargs
    ):
        super().__init__()
        language_model, tokenizer, llm_dmodel, resolved_name, resolved_layers = (
            load_language_backbone(
                backbone=backbone,
                model_name=backbone_name,
                num_layers=backbone_layers,
                backbone_init=backbone_init,
                prompt_mode=prompt_mode,
                train_backbone=train_backbone,
            )
        )
        self.llm_dmodel = llm_dmodel
        self.emb = Conv_MLP(n_feat, llm_dmodel, resid_pdrop=resid_pdrop)
        self.binary_indicator_embedding = nn.Linear(n_feat, llm_dmodel)
        self.gate_w1 = nn.Linear(llm_dmodel, llm_dmodel)
        self.gate_w2 = nn.Linear(llm_dmodel, llm_dmodel)
        self.gate_sigmoid = nn.Sigmoid()
        self.feature_projection = nn.Linear(llm_dmodel, llm_dmodel)
        self.ts_embed_dropout = nn.Dropout(0.3)
        self.n_channel = n_channel

        self.inverse = Conv_MLP(n_embd, n_feat, resid_pdrop=resid_pdrop)

        if conv_params is None or conv_params[0] is None:
            if n_feat < 32 and n_channel < 64:
                kernel_size, padding = 1, 0
            else:
                kernel_size, padding = 5, 2
        else:
            kernel_size, padding = conv_params

        self.combine_s = nn.Conv1d(n_embd, n_feat, kernel_size=kernel_size, stride=1, padding=padding,
                                   padding_mode='circular', bias=False)
        self.combine_m = nn.Conv1d(n_layer_dec, 1, kernel_size=1, stride=1, padding=0,
                                   padding_mode='circular', bias=False)

        self.encoder = Encoder(n_layer_enc, n_embd, n_heads, attn_pdrop, resid_pdrop, mlp_hidden_times, block_activate)
        self.pos_enc = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)

        self.decoder = Decoder(n_channel, n_feat, n_embd, n_heads, n_layer_dec, attn_pdrop, resid_pdrop, mlp_hidden_times,
                               block_activate, condition_dim=n_embd)
        self.pos_dec = LearnablePositionalEncoding(n_embd, dropout=resid_pdrop, max_len=max_len)


        self.backbone = str(backbone).lower()
        self.backbone_name = resolved_name
        self.backbone_layers = resolved_layers
        self.backbone_init = backbone_init
        self.prompt_mode = prompt_mode
        self.max_token_num = 17
        self.tokenizer = tokenizer

        # Keep this legacy attribute name so existing GPT-2 checkpoints retain
        # their state-dict keys.  It may now contain either GPT-2 or LLaMA.
        self.gpt2_tr = language_model

        if prompt_mode == 'learned_id':
            self.prompt_embedding = nn.Embedding(num_prompt_ids, llm_dmodel)

        # self.gpt2.h = self.gpt2.h[:2] # gpt_layer = 6

        self.gpt_out = nn.Linear(llm_dmodel, n_embd)
        self.gpt_out2 = nn.Linear(n_channel+self.max_token_num, n_channel)
        self.pad_token = nn.Parameter(torch.randn(1, 1, llm_dmodel), requires_grad=True)
#         self.gpt2_4_trend = GPT2Model.from_pretrained('gpt2-small', output_attentions=True, output_hidden_states=True)
#         self.gpt2_4_trend.h = self.gpt2_4_trend.h[:2] # gpt_layer = 6
#         self.gpt2_4_sea = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
#         self.gpt2_4_sea.h = self.gpt2_4_sea.h[:1] # gpt_layer = 6
#         if freeze and pretrain:
#             for i, (name, param) in enumerate(self.gpt2_4_sea.named_parameters()):
#                 if 'ln' in name or 'wpe' in name:
#                     param.requires_grad = True
#                 else:
#                     param.requires_grad = False


    def _prompt_embeddings(self, instruct, batch_size, device):
        if self.prompt_mode == 'none':
            return torch.empty(batch_size, 0, self.llm_dmodel, device=device)

        if self.prompt_mode == 'learned_id':
            prompt_id = int(instruct.item()) if torch.is_tensor(instruct) else int(instruct)
            prompt_ids = torch.tensor([prompt_id], dtype=torch.long, device=device)
            return self.prompt_embedding(prompt_ids).unsqueeze(0).repeat(batch_size, 1, 1)

        instruct_ids = self.tokenizer(
            str(instruct),
            return_tensors='pt',
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_token_num,
        ).input_ids.to(device)
        token_embeddings = self.gpt2_tr.get_input_embeddings()(instruct_ids)
        return token_embeddings.repeat(batch_size, 1, 1)

    def forward(self, instruct, input, t, mask = None, padding_masks=None, return_res=False):
        # print(' t is ', t) # [32, 24, 7]
        # print('input ',input.shape)
        b,tt,n = input.size()

        if mask is None:
            # print('mask none')
            mask = torch.ones((b, tt, n)).to(input.device)
        # print('mask shape is: ',mask.shape)    # b,tt,n
        # torch.sum(mask == 1, dim=1)
        # torch.sum(input, dim=1)
        # means = torch.sum(input, dim=1) / torch.sum(mask == 1, dim=1)
        # means = means.unsqueeze(1).detach()
        # input -= means
        # input = input.masked_fill(mask == 0, 0)
        # stdev = torch.sqrt(torch.sum(input * input, dim=1) /
        #                    torch.sum(mask == 1, dim=1) + 1e-5)
        # stdev = stdev.unsqueeze(1).detach()
        # input /= stdev



        emb = self.emb(input)


        # if mask!=None:

        #     mask_embed = self.binary_indicator_embedding(mask)
        #     gate = self.gate_sigmoid(self.gate_w1(emb) + self.gate_w2(mask_embed))
        #     emb = gate * emb + (1 - gate) * mask_embed
        #     emb = self.feature_projection(emb)
        #     emb = self.ts_embed_dropout(emb)
        #  print(' emb shape is ', emb.shape) # [32, 24, 768]

        instruct_embed = self._prompt_embeddings(instruct, emb.shape[0], input.device)
        # print('instruct_embed shape is ',instruct_embed.shape) # [32, 9, 768]
        inputs_embeds = torch.cat((instruct_embed, emb), dim=1)
        # print('inputs_embeds add emb shape is ',inputs_embeds.shape)  # [32, 24+9, 768]
#         # (2) to do : modify emb part to the unitime encoder+token+instruction part.





        backbone_output = self.gpt2_tr(inputs_embeds=inputs_embeds, use_cache=False)
        emb_out = hidden_states_from_output(backbone_output)
        b, token_num, _ = emb_out.shape
        pad_token_num = self.n_channel + self.max_token_num - token_num
        if pad_token_num > 0:
            p = self.pad_token.repeat(b, pad_token_num, 1)
            emb_out = torch.cat((emb_out, p), dim=1)

        #print('emb_out shape is ',emb_out.shape) # [32, 24+17, 768]
        emb = self.gpt_out(emb_out)
        #print('emb_out emb is ',emb.shape) ## [32, 24+17, 768] > [32, 24+17, 64]
        emb = self.gpt_out2(emb.permute(0,2,1)).permute(0,2,1)
        #print('emb is ',emb.shape) ## [32, 24+17, 768] > [32, 24, 64]





        inp_enc = self.pos_enc(emb)
        # print(' pos_enc inp_enc shape is ', inp_enc.shape) # [32, 24, 64]
        enc_cond = self.encoder(inp_enc, t, padding_masks=padding_masks)
       #  print(' enc_cond  is ', enc_cond.shape)  # [128, 24, 64]
        inp_dec = self.pos_dec(emb)
        #print(' pos_dec inp_enc shape is ', inp_dec.shape)   # [128, 24, 64]

        # (1) to do:  replace the encoder part refer to unitime and gpt4ts. no need to change decoder part because it's the key tech of Diffusion-TS
        # add the prompt token to a new dim, (different from unitime add token to spatial dim),   return dim without token---self.inverse part

        output, mean, trend, season = self.decoder(inp_dec, t, enc_cond, padding_masks=padding_masks)
       #  print(' decoder output shape is ', output.shape)  #  [128, 24, 64]
        #print(' decoder mean shape is ', mean.shape) # [128, 2, 7]
        #print(' decoder trend shape is ', trend.shape) # [128, 24, 7]
        #print(' decoder season shape is ', season.shape) # [128, 24, 64]

        res = self.inverse(output)
        #print(' inverse res shape is ', res.shape)  # [128, 24, 7]
        res_m = torch.mean(res, dim=1, keepdim=True)
        season_error = self.combine_s(season.transpose(1, 2)).transpose(1, 2) + res - res_m
        #print(' combine_s season_error shape is ', season_error.shape)  # # [128, 24, 7]
        trend = self.combine_m(mean) + res_m + trend
        #print(' combine_m trend shape is ', trend.shape)  # [128, 24, 7]
        if return_res:
            return trend, self.combine_s(season.transpose(1, 2)).transpose(1, 2), res - res_m


        # trend = trend * (stdev.repeat(1, trend.shape[1], 1))
        # trend = trend + (means.repeat(1, trend.shape[1], 1))
        # season_error = season_error * (stdev.repeat(1, season_error.shape[1], 1))
        # season_error = season_error + (means.repeat(1, season_error.shape[1], 1))


        return trend, season_error


if __name__ == '__main__':
    pass
