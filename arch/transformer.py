import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)
# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, num_patches, patch_dim, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 32, dropout = 0., emb_dropout = 0.):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, dim),
        )
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding1 = nn.Parameter(torch.randn(1,  1, dim))
        self.pos_embedding2_single = nn.Parameter(torch.randn(1,  1, dim))
        self.pos_embedding3_single = nn.Parameter(torch.randn(1,  1, dim))
        self.pos_embedding4_single = nn.Parameter(torch.randn(1,  1, dim))
        # self.pos_embedding1.to(self.device)
        # self.pos_embedding2_single.to(self.device)
        # self.pos_embedding3_single.to(self.device)
        # self.pos_embedding4_single.to(self.device)
        # self.pos_embedding2 = torch.tile(self.pos_embedding2_single,(1,3,1))
        # self.pos_embedding3 = torch.tile(self.pos_embedding3_single,(1,3,1))
        # self.pos_embedding4 = torch.tile(self.pos_embedding4_single,(1,3,1))
        # self.pos_embedding = torch.cat((self.pos_embedding1,self.pos_embedding2,self.pos_embedding3,self.pos_embedding4),dim=1)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, signal):
        x = self.to_patch_embedding(signal)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)

        self.pos_embedding2 = self.pos_embedding2_single.repeat((1,3,1))
        self.pos_embedding3 = self.pos_embedding3_single.repeat((1,3,1))
        self.pos_embedding4 = self.pos_embedding4_single.repeat((1,3,1))
        self.pos_embedding = torch.cat((self.pos_embedding1,self.pos_embedding2,self.pos_embedding3,self.pos_embedding4),dim=1)
        
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return self.mlp_head(x)

if __name__ == "__main__":
    model = ViT(
        num_patches = 9,
        patch_dim = 30,
        num_classes = 2,
        dim = 128,
        depth = 10,
        heads = 16,
        dim_head= 16,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.0
    )
    x = torch.rand((8,9,30))
    pred = model(x)
    print(pred.size())
