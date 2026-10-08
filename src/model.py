"""The compact SignVision CNN.

A small convolutional classifier that maps a single-channel 64x64 image to raw
class logits. Input is (N, 1, IMG_SIZE, IMG_SIZE); output is (N, NUM_CLASSES).
forward returns raw logits (no softmax) -- CrossEntropyLoss handles the softmax
at train time and src.app applies softmax explicitly at inference time.
"""

import torch.nn as nn

from src import config


class SignVisionCNN(nn.Module):
    """Three conv blocks (32/64/128 channels) with a 2-layer classifier head.

    Each MaxPool halves the spatial size: 64 -> 32 -> 16 -> 8, so the flattened
    feature map is 128 * 8 * 8.
    """

    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1 -> 32, two convs, then pool -> 32x32.
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32 -> 64, then pool -> 16x16.
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 64 -> 128, then pool -> 8x8.
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
