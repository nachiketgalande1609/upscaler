"""HAT (Hybrid Attention Transformer) architecture for image super-resolution.
Source: https://github.com/XPixelGroup/HAT  (CVPR 2023)
Copied here so the project has no dependency on the hat package.
"""

import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F

from basicsr.archs.arch_util import to_2tuple, trunc_normal_
from einops import rearrange


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ChannelAttention(nn.Module):
    def __init__(self, num_feat, squeeze_factor=16):
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        return x * self.attention(x)


class CAB(nn.Module):
    def __init__(self, num_feat, compress_ratio=3, squeeze_factor=30):
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def window_partition(x, window_size):
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)


def window_reverse(windows, window_size, h, w):
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, rpi, mask=None):
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        rpb = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1).permute(2, 0, 1).contiguous()
        attn = attn + rpb.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
        attn = self.attn_drop(self.softmax(attn))
        x = self.proj_drop(self.proj((attn @ v).transpose(1, 2).reshape(b_, n, c)))
        return x


class HAB(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 compress_ratio=3, squeeze_factor=30, conv_scale=0.01, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size),
                                    num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop)
        self.conv_scale = conv_scale
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x, x_size, rpi_sa, attn_mask):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        conv_x = self.conv_block(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous().view(b, h * w, c)
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if self.shift_size > 0 else x
        attn_mask_ = attn_mask if self.shift_size > 0 else None
        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, c)
        attn_windows = self.attn(x_windows, rpi=rpi_sa, mask=attn_mask_).view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)
        attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted_x
        x = shortcut + self.drop_path(attn_x.view(b, h * w, c)) + conv_x * self.conv_scale
        return x + self.drop_path(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        if self.norm:
            x = self.norm(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        return x.transpose(1, 2).contiguous().view(x.shape[0], self.embed_dim, x_size[0], x_size[1])


class OCAB(nn.Module):
    def __init__(self, dim, input_resolution, window_size, overlap_ratio, num_heads,
                 qkv_bias=True, qk_scale=None, mlp_ratio=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.overlap_win_size = int(window_size * overlap_ratio) + window_size
        self.norm1 = norm_layer(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.unfold = nn.Unfold(kernel_size=(self.overlap_win_size, self.overlap_win_size),
                                stride=window_size, padding=(self.overlap_win_size - window_size) // 2)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((window_size + self.overlap_win_size - 1) * (window_size + self.overlap_win_size - 1), num_heads))
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=nn.GELU)

    def forward(self, x, x_size, rpi):
        h, w = x_size
        b, _, c = x.shape
        shortcut = x
        x = self.norm1(x).view(b, h, w, c)
        qkv = self.qkv(x).reshape(b, h, w, 3, c).permute(3, 0, 4, 1, 2)
        q = qkv[0].permute(0, 2, 3, 1)
        kv = torch.cat((qkv[1], qkv[2]), dim=1)
        q_windows = window_partition(q, self.window_size).view(-1, self.window_size * self.window_size, c)
        kv_windows = self.unfold(kv)
        kv_windows = rearrange(kv_windows, 'b (nc ch owh oww) nw -> nc (b nw) (owh oww) ch',
                               nc=2, ch=c, owh=self.overlap_win_size, oww=self.overlap_win_size).contiguous()
        k_windows, v_windows = kv_windows[0], kv_windows[1]
        b_, nq, _ = q_windows.shape
        _, n, _ = k_windows.shape
        d = self.dim // self.num_heads
        q = q_windows.reshape(b_, nq, self.num_heads, d).permute(0, 2, 1, 3) * self.scale
        k = k_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)
        v = v_windows.reshape(b_, n, self.num_heads, d).permute(0, 2, 1, 3)
        attn = q @ k.transpose(-2, -1)
        rpb = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size * self.window_size, self.overlap_win_size * self.overlap_win_size, -1
        ).permute(2, 0, 1).contiguous()
        attn = self.softmax(attn + rpb.unsqueeze(0))
        attn_windows = (attn @ v).transpose(1, 2).reshape(b_, nq, self.dim)
        x = window_reverse(attn_windows.view(-1, self.window_size, self.window_size, self.dim),
                           self.window_size, h, w).view(b, h * w, self.dim)
        x = self.proj(x) + shortcut
        return x + self.mlp(self.norm2(x))


class AttenBlocks(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 compress_ratio, squeeze_factor, conv_scale, overlap_ratio,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            HAB(dim=dim, input_resolution=input_resolution, num_heads=num_heads,
                window_size=window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
                compress_ratio=compress_ratio, squeeze_factor=squeeze_factor, conv_scale=conv_scale,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)])
        self.overlap_attn = OCAB(dim=dim, input_resolution=input_resolution,
                                 window_size=window_size, overlap_ratio=overlap_ratio,
                                 num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 mlp_ratio=mlp_ratio, norm_layer=norm_layer)
        self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer) if downsample else None

    def forward(self, x, x_size, params):
        for blk in self.blocks:
            x = blk(x, x_size, params['rpi_sa'], params['attn_mask'])
        x = self.overlap_attn(x, x_size, params['rpi_oca'])
        if self.downsample:
            x = self.downsample(x)
        return x


class RHAG(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 compress_ratio, squeeze_factor, conv_scale, overlap_ratio,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super().__init__()
        self.residual_group = AttenBlocks(
            dim=dim, input_resolution=input_resolution, depth=depth, num_heads=num_heads,
            window_size=window_size, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor,
            conv_scale=conv_scale, overlap_ratio=overlap_ratio, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
            drop_path=drop_path, norm_layer=norm_layer, downsample=downsample,
            use_checkpoint=use_checkpoint)
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1) if resi_connection == '1conv' else nn.Identity()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=0, embed_dim=dim, norm_layer=None)
        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size,
                                          in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, x_size, params):
        return self.patch_embed(self.conv(self.patch_unembed(
            self.residual_group(x, x_size, params), x_size))) + x


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m += [nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1), nn.PixelShuffle(2)]
        elif scale == 3:
            m += [nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1), nn.PixelShuffle(3)]
        else:
            raise ValueError(f'Unsupported scale: {scale}')
        super().__init__(*m)


class HAT(nn.Module):
    """Hybrid Attention Transformer for Image Super-Resolution (CVPR 2023)."""

    def __init__(self, img_size=64, patch_size=1, in_chans=3, embed_dim=96,
                 depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6), window_size=7,
                 compress_ratio=3, squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 ape=False, patch_norm=True, use_checkpoint=False, upscale=2,
                 img_range=1., upsampler='', resi_connection='1conv', **kwargs):
        super().__init__()

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.overlap_ratio = overlap_ratio
        self.img_range = img_range
        self.upscale = upscale
        self.upsampler = upsampler

        if in_chans == 3:
            self.mean = torch.Tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.register_buffer('relative_position_index_SA', self._calc_rpi_sa())
        self.register_buffer('relative_position_index_OCA', self._calc_rpi_oca())

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=embed_dim, embed_dim=embed_dim,
                                      norm_layer=norm_layer if patch_norm else None)
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.patch_unembed = PatchUnEmbed(img_size=img_size, patch_size=patch_size,
                                          in_chans=embed_dim, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)

        if ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, self.patch_embed.num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.ape = ape

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList([
            RHAG(dim=embed_dim, input_resolution=(patches_resolution[0], patches_resolution[1]),
                 depth=depths[i], num_heads=num_heads[i], window_size=window_size,
                 compress_ratio=compress_ratio, squeeze_factor=squeeze_factor, conv_scale=conv_scale,
                 overlap_ratio=overlap_ratio, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                 qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate,
                 drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], norm_layer=norm_layer,
                 img_size=img_size, patch_size=patch_size, resi_connection=resi_connection)
            for i in range(len(depths))])

        self.norm = norm_layer(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1) if resi_connection == '1conv' else nn.Identity()

        if upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, 64, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, 64)
            self.conv_last = nn.Conv2d(64, in_chans, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _calc_rpi_sa(self):
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        rel = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += self.window_size - 1
        rel[:, :, 1] += self.window_size - 1
        rel[:, :, 0] *= 2 * self.window_size - 1
        return rel.sum(-1)

    def _calc_rpi_oca(self):
        ws = self.window_size
        we = ws + int(self.overlap_ratio * ws)
        ch = torch.arange(ws); cw = torch.arange(ws)
        ori = torch.flatten(torch.stack(torch.meshgrid([ch, cw])), 1)
        ch = torch.arange(we); cw = torch.arange(we)
        ext = torch.flatten(torch.stack(torch.meshgrid([ch, cw])), 1)
        rel = (ext[:, None, :] - ori[:, :, None]).permute(1, 2, 0).contiguous()
        rel[:, :, 0] += ws - we + 1
        rel[:, :, 1] += ws - we + 1
        rel[:, :, 0] *= ws + we - 1
        return rel.sum(-1)

    def _calc_mask(self, x_size):
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))
        h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x_size = (x.shape[2], x.shape[3])
        attn_mask = self._calc_mask(x_size).to(device=x.device, dtype=x.dtype)
        params = {
            'attn_mask': attn_mask,
            'rpi_sa': self.relative_position_index_SA,
            'rpi_oca': self.relative_position_index_OCA,
        }

        x = self.conv_first(x)
        feat = self.patch_embed(x)
        if self.ape:
            feat = feat + self.absolute_pos_embed
        feat = self.pos_drop(feat)
        for layer in self.layers:
            feat = layer(feat, x_size, params)
        feat = self.norm(feat)
        feat = self.patch_unembed(feat, x_size)

        x = self.conv_after_body(feat) + x
        x = self.conv_last(self.upsample(self.conv_before_upsample(x)))

        return x / self.img_range + self.mean
