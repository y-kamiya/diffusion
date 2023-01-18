import math
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kargs):
        return self.fn(x, *args, **kargs) + x


class Down(nn.Module):
    def __init__(self, dim):
        super(Down, self).__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, dim, dim_out):
        super(Up, self).__init__()
        self.conv_trans = nn.ConvTranspose2d(dim, dim_out, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.conv_trans(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)

    def forward(self, x):
        x = self.proj(x)
        return F.silu(self.norm(x))


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = None
        if time_emb_dim is not None:
            self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))

        self.block1 = Block(dim, dim_out, groups)
        self.block2 = Block(dim_out, dim_out, groups)
        self.conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            h = h + rearrange(time_emb, "b c -> b c 1 1")

        h = self.block2(h)
        return h + self.conv(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads=4, is_linear=False):
        super(MultiHeadAttention, self).__init__()
        self.scale = dim ** -0.5
        self.dim = dim
        self.n_heads = n_heads
        self.attention = self.linear_attention if is_linear else self.regular_attention

        hidden_dim = dim * n_heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.out_conv = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        batch_size, c, h, w = x.shape
        assert c == self.dim, 'dimension mismatched'

        def combine(x):
            x = x.transpose(1, 2)
            return x.contiguous().view(batch_size, -1, h, w)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.n_heads), qkv
        )

        output = self.attention(q, k, v)
        output = combine(output)

        return self.out_conv(output)

    def regular_attention(self, q, k, v):
        q = q * self.scale
        logit = torch.matmul(q, k.transpose(2, 3))
        weights = F.softmax(logit, dim=-1)
        return torch.matmul(weights, v)

    def linear_attention(self, q, k, v):
        q = F.softmax(q, dim=-2) * self.scale
        k = F.softmax(k, dim=-1)
        context = torch.matmul(k, v.transpose(2, 3))
        return torch.matmul(context, q)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        return self.fn(self.norm(x))


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Unet(nn.Module):
    def __init__(self, dim=32, dim_mults=(1, 2, 4, 8), in_channels=3):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, dim, 1, padding=0)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        in_out_size = len(in_out)

        time_dim = dim
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, groups=8)

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(dim, time_dim)
        )

        self.downs = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(in_out):
            self.downs.append(nn.ModuleList([
                resnet_block(d_in, d_out),
                resnet_block(d_out, d_out),
                Residual(PreNorm(d_out, MultiHeadAttention(d_out, is_linear=True))),
                Down(d_out) if i != in_out_size - 1 else nn.Identity(),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, MultiHeadAttention(mid_dim)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        self.ups = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(reversed(in_out)):
            self.ups.append(nn.ModuleList([
                resnet_block(d_out * 2, d_out),
                resnet_block(d_out, d_out),
                Residual(PreNorm(d_out, MultiHeadAttention(d_out, is_linear=True))),
                Up(d_out, d_in) if i != in_out_size - 1 else nn.Identity(),
            ]))

        self.out_conv = nn.Sequential(
            resnet_block(dim, dim),
            nn.Conv2d(dim, in_channels, 1)
        )

    def forward(self, x, t):
        x = self.in_conv(x)
        t = self.time_mlp(t)
        h = []

        for block1, block2, attn, down in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = down(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, up in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = up(x)

        return self.out_conv(x)
