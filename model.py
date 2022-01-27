from typing import Optional

import timm
import torch
from torch import nn


class ClsModel(nn.Module):
    def __init__(
        self,
        name: str,
        device: torch.device,
        pretrained: Optional[bool] = True,
        img_size: Optional[int] = 224,
        num_class: Optional[int] = 100,
    ):
        super(ClsModel, self).__init__()
        self.model = timm.create_model(
            name, pretrained=pretrained, img_size=img_size, num_classes=num_class
        )
        self.device = device
        self.model.to(device)
        self.score_fn = nn.Softmax(dim=1)

    def forward(self, x):
        return self.model(x.to(self.device))

    def predict(self, x):
        scores = self.score_fn(self.model(x.to(self.device)))
        return torch.argmax(scores, dim=1)
