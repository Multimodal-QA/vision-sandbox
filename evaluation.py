from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ClsModel


def evaluation(model: ClsModel, data, config, valid: Optional[bool] = False):
    model.eval()

    correct = torch.tensor(0, dtype=torch.int, device=config["device"])
    with torch.no_grad():
        for images, labels in tqdm(
            DataLoader(
                dataset=data, batch_size=config["valid" if valid else "eval"]["batch"]
            ),
            desc="Validation" if valid else "Evaluation",
        ):
            preds = model.predict(images)
            correct += torch.sum(preds == labels.to(config["device"]))

    print(f"top1 acc: {round(correct.item() / len(data), 8)}")
