import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn as nn
import numpy as np
from torch.nn.utils import weight_norm

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


class Transformer_freq(nn.Module):
    def __init__(self, *, num_patches, patch_dim, dim, depth, heads, mlp_dim, pool = 'cls', dim_head = 32, dropout = 0., emb_dropout = 0.):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, dim),
        )
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.pos_embedding1 = nn.Parameter(torch.randn(1,  1, dim))
        self.pos_embedding2_single = nn.Parameter(torch.randn(1,  1, dim))
        self.pos_embedding3_single = nn.Parameter(torch.randn(1,  1, dim))
        self.pos_embedding4_single = nn.Parameter(torch.randn(1,  1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim)
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
        return x

class CNN_spect(nn.Module):
    def __init__(self):
        super(CNN_spect, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 2, out_channels = 32, kernel_size = (3, 3), stride = 1, padding = 1, bias=True)
        self.conv2 = nn.Conv2d(in_channels = 32, out_channels=32, kernel_size=(3, 3), stride=1, padding=0, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=0, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1, bias=True)
        self.dropout3 = torch.nn.Dropout(0.5)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.bn3(x)
        x = x.view(x.size(0), x.size(1), -1)
        return x

class FCNet(nn.Module):
    """
    Simple class for multi-layer non-linear fully connect network
    Activate function: ReLU()
    """
    def __init__(self, dims, dropout=0.0, norm=True):
        super(FCNet, self).__init__()
        self.num_layers = len(dims) -1
        self.drop = dropout
        self.norm = norm
        self.main = nn.Sequential(*self._init_layers(dims))
 
    def _init_layers(self, dims):
        layers = []
        for i in range(self.num_layers):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            if self.norm:
                layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            else:
                layers.append(nn.Dropout(self.drop))
                layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        return layers

    def forward(self, x):
        return self.main(x)

class Attention_cross(nn.Module):
    def __init__(self, v_dim, q_dim, hid_dim, glimpses=1, dropout=0.2):
        super(Attention_cross, self).__init__()
        self.v_proj = FCNet([v_dim, hid_dim], dropout)
        self.q_proj = FCNet([q_dim, hid_dim], dropout)
        self.drop = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(hid_dim, glimpses), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        v_proj = self.v_proj(v)  # [batch, k, vdim]
        q_proj = self.q_proj(q).unsqueeze(1) # [batch, 1, qdim]
        logits = self.linear(self.drop(v_proj * q_proj))
        return nn.functional.softmax(logits, 1)

class Fusion_model(nn.Module):
    def __init__(self, num_classes):
        super(Fusion_model, self).__init__()
        self.num_classes = num_classes
        self.model_freq = Transformer_freq(
            num_patches = 9,
            patch_dim = 30,
            dim = 128,
            depth = 10,
            heads = 16,
            dim_head= 16,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.0
        )
        self.model_spect = CNN_spect()
        self.proj1 = nn.Linear(10*128, 128)
        self.proj2 = nn.Linear(78*128, 128)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = nn.LayerNorm(256)
        self.cross_att1 = Attention_cross(v_dim=128, q_dim=128, hid_dim=256, glimpses=1)
        self.cross_att2 = Attention_cross(v_dim=128, q_dim=128, hid_dim=256, glimpses=1)
        self.pooling1 = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, freq, spect):
        feat_freq = self.model_freq(freq)
        feat_spect = self.model_spect(spect)

        feat_freq_p = self.proj1(feat_freq.view(feat_freq.size(0),-1))
        feat_spect_p = self.proj2(feat_spect.view(feat_spect.size(0),-1))
   
        weight1 = self.cross_att1(feat_freq, feat_spect_p)
        feat_att_freq = torch.squeeze(torch.matmul(weight1.permute(0,2,1), feat_freq),dim=1)
        weight2 = self.cross_att2(feat_spect.permute(0,2,1), feat_freq_p)
        feat_att_spect = torch.squeeze(torch.matmul(weight2.permute(0,2,1), feat_spect.permute(0,2,1)),dim=1)
        feat_cat = torch.cat((feat_att_freq, feat_att_spect), dim=1)
        x = self.norm1(feat_cat)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits

if __name__ == "__main__":
    model = Fusion_model(2)
    spects = torch.rand(size=(8, 2, 33, 61))
    x2 = torch.rand((8,9,30))
    logits = model(x2, spects)
    print(logits.size())