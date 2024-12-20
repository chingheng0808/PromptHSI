import torch
from torch import nn
from torch.nn import functional as F
from utils.modules import (
    TransformerBlock,
    Attention_spatial,
    LayerNorm,
    Fusion_Embed,
    Cross_attention,
    FeatureWiseAffine,
    initialize_weights
)
from utils.DRCT import RDG, PatchUnEmbed, PatchEmbed
from utils.lossfunc import HyperspectralSWTLoss, SAMLoss, BandWiseMSE
from timm.models.layers import to_2tuple, trunc_normal_

BatchNorm2d = nn.BatchNorm2d

class emptyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

###########################################
class SpectralWiseAttention(nn.Module):
    def __init__(self, dim, bias=False):
        super(SpectralWiseAttention, self).__init__()
        self.sigma = nn.Parameter(torch.ones(1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.linear = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.view(b, c, -1).permute(0, 2, 1)  # b, h*w, c
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = torch.nn.functional.normalize(q, dim=1)
        k = torch.nn.functional.normalize(k, dim=1)

        attn = (k.transpose(-2, -1) @ q) * self.sigma
        attn = attn.softmax(dim=-1)
        out = self.linear(v @ attn).permute(0, 2, 1).view(b, c, h, w)

        return out


class SpectralAttentionBlock(nn.Module):
    def __init__(self, dim, bias=False, LayerNorm_type="WithBias"):
        super(SpectralAttentionBlock, self).__init__()
        if LayerNorm_type is None:
            self.norm = emptyModule()
        else:
            self.norm = LayerNorm(dim, LayerNorm_type=LayerNorm_type)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, stride=1, bias=False)
        self.specatt = SpectralWiseAttention(dim, bias)

    def forward(self, x):
        res = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.specatt(x)
        x = self.conv2(x)
        x = x + res
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_channel, embeding_dim, bias):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, embeding_dim, 5, 1, 2, bias=bias)
        self.conv2 = self.depwiseSepConv(embeding_dim, embeding_dim * 2**1, 5, bias)
        self.conv3 = self.depwiseSepConv(
            embeding_dim * 2**1, embeding_dim * 2**2, 3, bias
        )
        self.conv4 = self.depwiseSepConv(
            embeding_dim * 2**2,
            embeding_dim * 2**3,
            3,
            bias,
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return x1, x2, x3, x4

    def depwiseSepConv(self, in_dim, out_dim, ker_sz, bias=False):
        depwiseConv = nn.Conv2d(
            in_dim, in_dim, ker_sz, 2, ker_sz // 2, groups=in_dim, bias=bias
        )
        ptwiseConv = nn.Conv2d(in_dim, out_dim, 1, 1, 0, bias=bias)
        bn = BatchNorm2d(out_dim)
        relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        return nn.Sequential(depwiseConv, ptwiseConv, bn, relu)

'''
@misc{hsu2024realtimecompressedsensingjoint,
      title={Real-Time Compressed Sensing for Joint Hyperspectral Image Transmission and Restoration for CubeSat}, 
      author={Chih-Chung Hsu and Chih-Yu Jian and Eng-Shen Tu and Chia-Ming Lee and Guan-Lin Chen},
      year={2024},
      eprint={2404.15781},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.15781}, 
}
'''
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf, gc=32, bias=False, groups=4):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias, groups=groups)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias, groups=groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5*0.2 + x

'''
@misc{hsu2024drctsavingimagesuperresolution,
      title={DRCT: Saving Image Super-resolution away from Information Bottleneck}, 
      author={Chih-Chung Hsu and Chia-Ming Lee and Yi-Shiuan Chou},
      year={2024},
      eprint={2404.00722},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.00722}, 
}
'''
class RDGsBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        img_size,
        num_heads,
        window_size,
        patch_size,
        num_layers=1,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        gc=32,
        mlp_ratio=4.0,
        drop_path=0.0,
    ):
        super(RDGsBlock, self).__init__()

        self.ape = ape
        self.window_size = window_size
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.patch_norm = patch_norm

        self.conv = nn.Conv2d(
            in_dim, in_dim // 4, kernel_size=1, stride=1, bias=False, groups=in_dim // 4
        )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_dim // 4,
            embed_dim=in_dim // 4,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_dim // 4,
            embed_dim=in_dim // 4,
            norm_layer=norm_layer if self.patch_norm else None,
        )

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, in_dim // 4)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                RDG(
                    dim=in_dim // 4,
                    input_resolution=(patches_resolution[0], patches_resolution[1]),
                    num_heads=num_heads,
                    window_size=window_size,
                    depth=0,
                    shift_size=window_size // 2,
                    drop_path=drop_path,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    gc=gc,
                    img_size=img_size,
                    patch_size=patch_size,
                )  # with 5 swin layers
            )

        self.norm = norm_layer(in_dim // 4)
        self.conv_up = nn.Conv2d(
            in_dim // 4, in_dim, kernel_size=1, stride=1, bias=False
        )

    def forward(self, x):
        x_size = (x.shape[2], x.shape[3])

        x = self.conv(x)
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)
        x = self.conv_up(x)  # + 0.5 * x

        return x  # out_channel equals the same in_chammel of x

class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        patch_size,
        img_size=(224, 224),
        num_layers=(2, 1),
        bias=False,
        upsample=True,
    ):
        super(DecoderBlock, self).__init__()
        self.upsample = upsample
        self.num_layers = num_layers

        if self.num_layers[0] > 0:
            self.conv_spa_1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias, groups=1)
        if self.num_layers[1] > 0:
            self.conv_spe_1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=bias, groups=1)

        if self.num_layers[0]>0 and self.num_layers[1]>0:
            self.cross_att = Cross_attention(dim, norm_groups=dim // 4)
        if self.num_layers[0] > 0 or self.num_layers[1] > 0:
            self.feature_fusion = Fusion_Embed(embed_dim=dim)
        
        self.attention_spatial = Attention_spatial(
            dim, n_head=num_heads // 2, norm_groups=dim // 4
        )
        self.attention_spectral = SpectralAttentionBlock(
            dim, bias=bias, LayerNorm_type="WithBias"
        )
        self.prompt_guidance = FeatureWiseAffine(in_channels=512, out_channels=dim)

        if self.num_layers[0] > 0:
            self.spatial_branch = RDGsBlock(
                dim,
                num_layers=num_layers[0],
                img_size=img_size[0],
                num_heads=num_heads,
                window_size=window_size,
                patch_size=patch_size,
            )
        if self.num_layers[1] > 0:
            self.spectral_branch = nn.Sequential(
            *[
               ResidualDenseBlock_5C(
                   dim
               )
               for _ in range(self.num_layers[1])
            ]
            )

        self.upconv = nn.Conv2d(dim, dim // 2, 3, 1, 1, bias=bias, groups=dim // 2)
        self.HRconv = nn.Conv2d(dim // 2, dim // 2, 1, 1, 0, bias=False, groups=1)

        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x, text_emb):
        if self.num_layers[0] > 0 and self.num_layers[1] > 0:
            fea1 = self.prompt_guidance(self.conv_spa_1(x), text_emb)
            fea2 = self.prompt_guidance(self.conv_spe_1(x), text_emb)
        elif self.num_layers[0] > 0:
            fea1 = self.prompt_guidance(self.conv_spa_1(x), text_emb)
            fea2 = x
        elif self.num_layers[1] > 0:
            fea1 = x
            fea2 = self.prompt_guidance(self.conv_spe_1(x), text_emb)

        if self.num_layers[0] > 0:
            fea1 = self.spatial_branch(fea1)
        if self.num_layers[1] > 0:
            fea2 = self.spectral_branch(fea2)

        if self.num_layers[0] > 0 and self.num_layers[1] > 0:
            fea1, fea2 = self.cross_att(fea1, fea2)
        
        if self.num_layers[0] > 0 or self.num_layers[1] > 0:
            x = self.feature_fusion(fea1, fea2)
        
        x = self.attention_spectral(x)
        x = self.attention_spatial(x)

        if self.upsample:
            x = self.lrelu(
                self.upconv(F.interpolate(x, scale_factor=2, mode="bilinear"))
            )
            x = self.HRconv(x)

        return x


class PromptHSI(nn.Module):
    def __init__(
        self,
        img_size,
        in_channel=172,
        embeding_dim=64,
        num_blocks_tf=2,
        num_layers=(2, 1),
        num_heads=8,
        window_size=(7, 7, 7),
        patch_size=(4, 4, 4),
        bias=False,
        LayerNorm_type="WithBias",
    ):
        super(PromptHSI, self).__init__()

        self.encoder = Encoder(in_channel, embeding_dim, bias=True)
        self.conv_tail = nn.Conv2d(2 * embeding_dim, in_channel, 1, 1, 0, bias=bias)

        self.decoder4 = DecoderBlock(
            embeding_dim * 2**3,
            num_heads,
            window_size[0],
            patch_size[0],
            to_2tuple(img_size[0] // 8),
            bias=bias,
            num_layers=num_layers,
            upsample=True,
        )
        self.decoder3 = DecoderBlock(
            2 * embeding_dim * 2**1,
            num_heads,
            window_size[1],
            patch_size[1],
            to_2tuple(img_size[0] // 4),
            bias=bias,
            num_layers=num_layers,
        )
        self.decoder2 = DecoderBlock(
            embeding_dim * 2**1,
            num_heads // 2,
            window_size[2],
            patch_size[2],
            to_2tuple(img_size[0] // 2),
            bias=bias,
            num_layers=num_layers,
            upsample=True,
        )

        # Enhancement block
        self.enhance = nn.Sequential(
            *[
                TransformerBlock(
                    dim=2 * embeding_dim,
                    num_heads=num_heads // 2,
                    ffn_expansion_factor=2,
                    bias=bias,
                    LayerNorm_type=LayerNorm_type,
                )
                for i in range(num_blocks_tf)
            ]
        )
        # Enhancement block end

        self.conv_a3 = nn.Conv2d(
            2 * embeding_dim * 2**2, embeding_dim * 2**2, 1, 1, bias=bias
        )
        self.conv_a2 = nn.Conv2d(
            2 * embeding_dim * 2**1, embeding_dim * 2**1, 1, 1, bias=bias
        )
        self.conv_a1 = nn.Conv2d(2 * embeding_dim, 2 * embeding_dim, 1, 1, bias=bias)

        ## Metrics
        self.L1Loss = torch.nn.L1Loss()
        self.sam_loss = SAMLoss()
        self.BandWiseMSE = BandWiseMSE()
        self.Waveletloss = HyperspectralSWTLoss()

    def forward(self, x, x_gt, text_emb):
        x1, x2, x3, x4 = self.encoder(x)  # for 112 size hsi: 112, 56, 28, 14
        x = self.decoder4(x4, text_emb)
        x = self.conv_a3(torch.concat((x, x3), 1))
        x = self.decoder3(x, text_emb)
        x = self.conv_a2(torch.concat((x, x2), 1))
        x = self.decoder2(x, text_emb)
        x = self.conv_a1(torch.concat((x, x1), 1))
        x = self.enhance(x) + x
        x = self.conv_tail(x)

        loss1 = torch.unsqueeze(self.L1Loss(x, x_gt), 0)
        loss2 = torch.unsqueeze(self.BandWiseMSE(x, x_gt), 0)
        loss3 = torch.unsqueeze(self.sam_loss(x, x_gt), 0)
        loss4 = torch.unsqueeze(self.Waveletloss(x, x_gt), 0)

        return x, x1, x2, x3, x4, loss1, loss2, loss3, loss4

