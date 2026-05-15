"""
SpinalNet classification head.

Inspired by the hierarchical, segmented processing of the human spinal cord
(Kabir et al., 2020 — arXiv:2007.03347). The input features are divided into
two halves that cycle through four spinal layers, each receiving the current
input segment concatenated with the accumulated output of the previous layer.
This improves gradient flow and reduces the vanishing gradient problem.
"""

import torch
import torch.nn as nn


class SpinalNet(nn.Module):
    def __init__(self, in_features: int, num_classes: int, layer_width: int = 512, dropout: float = 0.5):
        super().__init__()

        if in_features % 2 != 0:
            raise ValueError(f"in_features must be even, got {in_features}")

        self.half = in_features // 2
        self.layer_width = layer_width

        def spinal_layer(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_dim, layer_width),
                nn.LeakyReLU(negative_slope=0.01, inplace=True),
            )

        # Layer 1: first half of input only
        self.fc1 = spinal_layer(self.half)
        # Layers 2-4: alternating halves + previous layer output
        self.fc2 = spinal_layer(self.half + layer_width)
        self.fc3 = spinal_layer(self.half + layer_width)
        self.fc4 = spinal_layer(self.half + layer_width)

        self.classifier = nn.Linear(4 * layer_width, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1_in = x[:, : self.half]
        x2_in = x[:, self.half :]

        out1 = self.fc1(x1_in)
        out2 = self.fc2(torch.cat([x2_in, out1], dim=1))
        out3 = self.fc3(torch.cat([x1_in, out2], dim=1))
        out4 = self.fc4(torch.cat([x2_in, out3], dim=1))

        return self.classifier(torch.cat([out1, out2, out3, out4], dim=1))
