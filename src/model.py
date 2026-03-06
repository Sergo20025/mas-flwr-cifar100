from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.dropout_rate = dropout_rate
        self.equal_in_out = in_planes == out_planes and stride == 1

        if not self.equal_in_out:
            self.shortcut = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(x))
        shortcut = x if self.equal_in_out else self.shortcut(out)

        out = self.conv1(out)
        out = self.relu2(self.bn2(out))

        if self.dropout_rate > 0.0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        out = self.conv2(out)
        out = out + shortcut
        return out


class NetworkBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_planes: int,
        out_planes: int,
        block: type[BasicBlock],
        stride: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(
                block(
                    in_planes=in_planes if i == 0 else out_planes,
                    out_planes=out_planes,
                    stride=stride if i == 0 else 1,
                    dropout_rate=dropout_rate,
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        dropout_rate: float = 0.3,
        num_classes: int = 100,
    ) -> None:
        super().__init__()

        if (depth - 4) % 6 != 0:
            raise ValueError("WideResNet depth should satisfy (depth - 4) % 6 == 0")

        num_layers_per_block = (depth - 4) // 6
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(
            3,
            channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.block1 = NetworkBlock(
            num_layers=num_layers_per_block,
            in_planes=channels[0],
            out_planes=channels[1],
            block=BasicBlock,
            stride=1,
            dropout_rate=dropout_rate,
        )
        self.block2 = NetworkBlock(
            num_layers=num_layers_per_block,
            in_planes=channels[1],
            out_planes=channels[2],
            block=BasicBlock,
            stride=2,
            dropout_rate=dropout_rate,
        )
        self.block3 = NetworkBlock(
            num_layers=num_layers_per_block,
            in_planes=channels[2],
            out_planes=channels[3],
            block=BasicBlock,
            stride=2,
            dropout_rate=dropout_rate,
        )

        self.bn = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(num_classes: int = 100) -> nn.Module:
    return WideResNet(
        depth=28,
        widen_factor=10,
        dropout_rate=0.3,
        num_classes=num_classes,
    )