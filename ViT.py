import torch 
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, in_channels = 3):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.embeding = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
    def forward(self, x):
        x = self.unfold(x)
        x = x.transpose(1,2)
        x = self.embeding(x)
        return x
    
class ClassToken(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim = 1)
        return x
    
class PositionEmbeding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True)
    def forward(self, x):
        x = x + self.pos_embed
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.msa = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True, bias=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x_norm = self.ln1(x)
        attn = self.msa(x_norm, x_norm, x_norm)[0]
        x = attn + x
        x_norm = self.ln2(x)
        x = self.mlp(x_norm) + x
        return x 
    
class Classifier(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
    def forward(self, x):
        y = self.fc(x)
        return y
    
class ViT(nn.Module):
    def __init__(self, img_size, in_chanels, patch_size, embed_dim, num_encoders, num_heads, hidden_dim, dropout, num_classes):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        self.pe = PatchEmbedding(img_size, patch_size, embed_dim, in_chanels)
        self.cls_token = ClassToken(embed_dim)
        self.pos = PositionEmbeding(num_patches, embed_dim)
        self.trans = nn.Sequential(*[TransformerEncoder(embed_dim, num_heads, hidden_dim, dropout) for _ in range(num_encoders)])
        self.ln = nn.LayerNorm(embed_dim)
        self.cls = Classifier(embed_dim, num_classes)
    def forward(self, x):
        x = self.pe(x)
        x = self.cls_token(x)
        x = self.pos(x)
        x = self.trans(x)[:,0,:].squeeze(1)
        x = self.ln(x)
        x = self.cls(x)
        return x

class ViT_Extractor(nn.Module):
    def __init__(self, vit_model):
        super().__init__()
        self.pe = vit_model.pe
        self.cls_token = vit_model.cls_token
        self.pos = vit_model.pos
        self.trans = vit_model.trans
        self.ln = vit_model.ln

    def forward(self, x):
        x = self.pe(x)
        x = self.cls_token(x)
        x = self.pos(x)
        x = self.trans(x)[:,0,:].squeeze(1)
        x = self.ln(x)
        return x