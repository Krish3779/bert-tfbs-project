# ========================================================================
# [RUN GUIDE]
# script/model_resnet.py  -  BERT-TFBS with the CNN front end replaced by a ResNet
# (library file: do NOT run it directly; chosen with --arch resnet)
#
# Only the CNN front end (conv1, parallel branches, conv3) is replaced. CBAM and the
# output module are inherited unchanged from script/model.py.
#
#   Quick shape check (from ~/BERT-TFBS, container up):
#      docker compose exec bert-tfbs python -c "import torch; from script.model_resnet import ResNetCNNNET; print(ResNetCNNNET(30).eval()(torch.randn(4,30,768)).shape)"
#      expected: torch.Size([4, 2])
#
#   Train / test:
#      docker compose exec -d bert-tfbs sh -c "python -u train_metrics.py --arch resnet --tag resnet --seed 41 > /workspace/results_resnet_41.log 2>&1"
#      docker compose exec bert-tfbs python test.py --arch resnet --weights model_resnet_seed41.pth --out results/test_resnet_seed41.csv
#
#   Depth: n_blocks in Bert_Blend_CNN.__init__ (default 2). Use a new --tag when you change it.
# ========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .CBAM import ChannelGate, SpatialGate


class ResBlock1D(nn.Module):
    """conv-BN-GELU-dropout-conv-BN + skip connection."""

    def __init__(self, in_ch, out_ch, stride=1, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class ResNetNET(nn.Module):
    """Drop-in replacement for CNNNET: (B, input_channel, 768) -> (B, 2) class probabilities."""

    def __init__(self, input_channel, base=64, blocks=(2, 2, 2), use_cbam=True):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channel, base, 7, padding=3, bias=False),
            nn.BatchNorm1d(base),
            nn.GELU())
        layers, in_ch = [], base
        for i, n in enumerate(blocks):
            out_ch = base * 2 ** i
            for j in range(n):
                stride = 2 if (i > 0 and j == 0) else 1
                layers.append(ResBlock1D(in_ch, out_ch, stride))
                in_ch = out_ch
        self.layers = nn.Sequential(*layers)
        self.use_cbam = use_cbam
        if use_cbam:
            self.ChannelGate = ChannelGate(gate_channels=in_ch, reduction_ratio=16,
                                           pool_types=['avg', 'max'])
            self.SpatialGate = SpatialGate()
        self.head = nn.Sequential(
            nn.Linear(2 * in_ch, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2))

    def forward(self, x):
        x = self.layers(self.stem(x))
        if self.use_cbam:
            x = self.SpatialGate(self.ChannelGate(x))
        x = torch.cat([x.mean(dim=2), x.amax(dim=2)], dim=1)  # global avg + max pooling
        return F.softmax(self.head(x), dim=1)  # softmax kept to match the original code


class Bert_Blend_CNN(nn.Module):
    """Same name and signature as the original, so the scripts work unchanged."""

    def __init__(self, input_channel, transpose=False, use_cbam=True):
        super().__init__()
        self.bert = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        for p in self.bert.parameters():
            p.requires_grad = True
        self.transpose = transpose
        # transpose=False: tokens are channels (original behaviour)
        # transpose=True : the 768 hidden dims are channels, convolution runs along the sequence
        self.model = ResNetNET(768 if transpose else input_channel, use_cbam=use_cbam)

    def forward(self, X):
        h = self.bert(X)[0]
        if self.transpose:
            h = h.transpose(1, 2)
        return self.model(h)
