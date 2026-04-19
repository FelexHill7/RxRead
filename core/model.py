import torch
import torch.nn as nn
import torchvision.models as models


# ── Gated Convolution Block ──────────────────────────────────────────────────
class GatedConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.gate = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)

    def forward(self, x):
        feat = self.conv(x)
        gate = torch.sigmoid(self.gate(x))
        return feat * gate


class ResNetCRNN(nn.Module):
    def __init__(self, num_classes, dropout=0.3):
        super().__init__()

        # ── Load pretrained ResNet ───────────────────────────────────────────
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # 🔥 KEEP pretrained weights — convert grayscale → 3-channel instead
        self.input_adapter = nn.Conv2d(1, 3, kernel_size=1)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

        self.layer3 = resnet.layer3
        self._modify_stride(self.layer3, (2, 1))

        self.layer4 = resnet.layer4
        self._modify_stride(self.layer4, (2, 1))

        self.pool = nn.AdaptiveAvgPool2d((1, None))

        # ── 🔥 ADD GATED CONV (key upgrade) ────────────────────────────────
        self.gated = GatedConv(512, 512)

        # ── Feature projection (VERY important) ────────────────────────────
        self.proj = nn.Linear(512, 256)

        # ── BiLSTM ────────────────────────────────────────────────────────
        self.rnn = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        # ── Output ────────────────────────────────────────────────────────
        self.fc = nn.Linear(512, num_classes)

    @staticmethod
    def _modify_stride(layer, new_stride):
        block = layer[0]

        old_conv = block.conv1
        new_conv = nn.Conv2d(
            old_conv.in_channels,
            old_conv.out_channels,
            old_conv.kernel_size,
            stride=new_stride,
            padding=old_conv.padding,
            bias=False
        )
        new_conv.weight.data.copy_(old_conv.weight.data)
        block.conv1 = new_conv

        if block.downsample is not None:
            old_ds = block.downsample[0]
            new_ds = nn.Conv2d(
                old_ds.in_channels,
                old_ds.out_channels,
                old_ds.kernel_size,
                stride=new_stride,
                bias=False
            )
            new_ds.weight.data.copy_(old_ds.weight.data)
            block.downsample[0] = new_ds

    def forward(self, x):
        # ── Convert grayscale → 3-channel ────────────────────────────────
        x = self.input_adapter(x)

        # ── CNN backbone ────────────────────────────────────────────────
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.gated(x)   # 🔥 NEW

        x = self.pool(x)    # (B, 512, 1, W)
        x = x.squeeze(2)    # (B, 512, W)
        x = x.permute(0, 2, 1)  # (B, W, 512)

        # ── Feature compression ────────────────────────────────────────
        x = self.proj(x)

        # ── Sequence modeling ──────────────────────────────────────────
        x, _ = self.rnn(x)
        x = self.dropout(x)

        # ── Output ─────────────────────────────────────────────────────
        x = self.fc(x)
        x = x.permute(1, 0, 2)  # (T, B, C)

        return x