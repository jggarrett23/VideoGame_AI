import torch.nn as nn
from torch import Tensor
import torch
import numpy as np
from transformers import DeiTFeatureExtractor, DeiTModel
import matplotlib.pyplot as plt


def PositionEmbedding(seq_len, emb_size):
    # each patch in the image is treated as a token. class token is a specific token appended to seq of patches
    # token serves as a representative of entire image. token embeds info about global context of image
    # and allows model to capture relationships between patches
    embeddings = torch.ones(seq_len, emb_size)
    for i in range(seq_len):
        for j in range(emb_size):
            if j % 2:
                embeddings[i][j] = np.cos(i / 1000 ** ((j - 1) / emb_size))
            else:
                embeddings[i][j] = np.sin(i / 1000 ** (j / emb_size))
    return torch.Tensor(embeddings)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size=84):
        super().__init__()
        self.patch_size = patch_size
        self.embed_size = emb_size
        self.projection = nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.token_seq_len = (img_size // patch_size) ** 2 + 1
        self.pos_embed = nn.Parameter(PositionEmbedding(self.token_seq_len, emb_size))

    def forward(self, X: Tensor) -> Tensor:
        b, _, _, _ = X.size()
        X = self.projection(X)
        X = X.permute(0, 2, 3, 1).contiguous().view(b, -1, self.embed_size)
        cls_token = self.cls_token.expand(b, -1, -1)
        X = torch.cat([cls_token, X], dim=1)
        out = X + self.pos_embed
        return out


class MultiHead(nn.Module):
    # Vision Transformer
    def __init__(self, emb_size, num_head=2):
        super().__init__()
        self.emb_size = emb_size
        self.num_head = num_head
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)
        self.query = nn.Linear(emb_size, emb_size)
        self.dp = nn.Dropout(0.1)
        self.emb_per_head = emb_size // num_head
        # assert 128 % emb_size == 0

    def forward(self, X: Tensor) -> Tensor:
        b, n, e = X.size()
        k = self.key(X).view(b, n, self.num_head, self.emb_per_head).permute(0, 2, 1, 3)
        q = self.value(X).view(b, n, self.num_head, self.emb_per_head).permute(0, 2, 1, 3)
        v = self.query(X).view(b, n, self.num_head, self.emb_per_head).permute(0, 2, 1, 3)

        wei = q @ k.transpose(3, 2) / self.num_head ** 0.5
        wei = nn.functional.softmax(wei, dim=2)
        wei = self.dp(wei)

        out = wei @ v
        out = out.permute(0, 2, 1, 3).contiguous().view(b, n, -1)

        return out


class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_head=2):
        super().__init__()
        self.att = MultiHead(emb_size, num_head)
        self.ll = nn.LayerNorm(emb_size)
        self.dp = nn.Dropout(0.1)
        self.FC = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.Linear(4 * emb_size, emb_size)
        )

    def forward(self, X: Tensor) -> Tensor:
        X = X + self.dp(self.att(self.ll(X)))
        out = X + self.dp(self.FC(self.ll(X)))
        return out


class ViT(nn.Module):
    def __init__(self, num_layers, in_channels=4, img_size=128, emb_size=768, patch_size=16,
                 num_head=2, num_class=18):
        super().__init__()
        self.attention = nn.Sequential(
            *[TransformerEncoderBlock(emb_size, num_head) for _ in range(num_layers)]
        )
        self.patchemb = PatchEmbedding(in_channels=in_channels,
                                       patch_size=patch_size, emb_size=emb_size, img_size=img_size)
        self.FC1 = nn.Linear(emb_size, num_class)
        self.FC2 = nn.Linear(emb_size, 1)
        self.layer_norm = nn.LayerNorm(emb_size)

    def forward(self, X: Tensor) -> Tensor:
        embeddings = self.patchemb(X)
        X = self.attention(embeddings)
        out1 = self.FC1(X[:, -1, :])
        # prevent key press duration from being less than 0.1
        x_norm = self.layer_norm(X[:, -1, :])
        out2 = torch.clamp(self.FC2(x_norm), min=0.1, max=6)

        return out1, out2


class PreTrained_DeiTModel(nn.Module):
    def __init__(self, model_name='facebook/deit-base-patch16-224', in_channels=4, num_classes=18):
        super(PreTrained_DeiTModel, self).__init__()
        # Load Pretrained model
        self.feature_extractor = DeiTModel.from_pretrained(model_name)
        self.feature_extractor.eval()  # ensure model is in eval mode

        # Freeze DeiT parameters
        n_feature_extractor_prams = len(list(self.feature_extractor.parameters()))
        cnt = 0
        for param in self.feature_extractor.parameters():
            if cnt < n_feature_extractor_prams:
                param.requires_grad = False
                cnt += 1

        # Extract hidden size from Deit model
        hidden_size = self.feature_extractor.config.hidden_size

        # Pretrained model requires 3 color channels
        self.channel_projection = nn.Conv2d(in_channels=in_channels, out_channels=3, kernel_size=1)

        # Classifier layers
        self.fc1 = nn.Linear(hidden_size, num_classes)  # Action prediction
        self.fc2 = nn.Linear(hidden_size, 1)  # Key press duration prediction
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, X: Tensor) -> Tensor:
        X = self.channel_projection(X)
        with torch.no_grad():
            outputs = self.feature_extractor(pixel_values=X)
            cls_token_features = outputs.last_hidden_state[:, 0, :]  # CLS token features

        action_logits = self.fc1(cls_token_features)
        norm_features = self.layer_norm(cls_token_features)
        duration = torch.clamp(self.fc2(norm_features), min=0.1, max=6)

        return action_logits, duration


class cnn_fc(nn.Module):
    def __init__(self, in_channels=3, img_shape=(128, 128), num_classes=18, ):
        super(cnn_fc, self).__init__()
        self.img_shape = img_shape
        self.in_channels = in_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=32,
                kernel_size=5
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=1
            ),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Flatten()
        )
        conv_out = self._get_conv_out()
        self.fc = nn.Linear(in_features=conv_out.shape[-1], out_features=num_classes)
        self.duration_fc = nn.Linear(in_features=conv_out.shape[-1], out_features=1)

    def _get_conv_out(self):
        X = torch.zeros([1, self.in_channels, self.img_shape[0], self.img_shape[1]])
        out = self.cnn(X)
        return out

    def forward(self, X: Tensor) -> Tensor:
        # add noise to input during training (data augmentation)
        if self.cnn.training:
            X += torch.randn(X.size(), device=X.device)
        cnn_out = self.cnn(X)
        action_logits = self.fc(cnn_out)
        duration_preds = torch.clamp(self.duration_fc(cnn_out), min=0.1, max=6)
        return action_logits, duration_preds


class dueling_cnn(nn.Module):
    """Dueling DQN (Wang et al. 2016) with stride-based CNN reduction.

    Compared to cnn_fc: stride/kernel sizes follow the Nature DQN paper, cutting
    the flattened feature vector from ~1.9M to ~9K, and the FC head is split into
    separate value and advantage streams so the network can learn state value
    independently of per-action advantage.
    """

    def __init__(self, in_channels: int = 4, img_shape: tuple = (128, 128), num_classes: int = 18):
        super().__init__()
        self.img_shape = img_shape
        self.in_channels = in_channels

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Flatten(),
        )
        conv_out_size = self._get_conv_out().shape[-1]

        self.shared_fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
        )

        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Advantage stream: estimates A(s, a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

        # Key-press duration head
        self.duration_fc = nn.Linear(512, 1)

    def _get_conv_out(self) -> Tensor:
        X = torch.zeros(1, self.in_channels, self.img_shape[0], self.img_shape[1])
        return self.cnn(X)

    def forward(self, X: Tensor) -> tuple[Tensor, Tensor]:
        if self.training:
            X = X + torch.randn(X.size(), device=X.device)
        features = self.shared_fc(self.cnn(X))

        value = self.value_stream(features)                          # (B, 1)
        advantage = self.advantage_stream(features)                  # (B, num_classes)
        # Q(s,a) = V(s) + A(s,a) - mean_a[A(s,a)]
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        duration = torch.clamp(self.duration_fc(features), min=0.1, max=6)
        return q_values, duration
