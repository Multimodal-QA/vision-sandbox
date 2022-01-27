import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from evaluation import evaluation
from model import ClsModel


def train(model: ClsModel, train_data, config):
    if config["valid"]["do_valid"]:
        print("validation will be proceeded during training")
        assert config["valid"]["steps"] > 0

    if config["valid"]["train_split"]:
        print("split data ...")
        assert 0 < config["valid"]["rate"] < 1
        n_valid = int(len(train_data) * config["valid"]["rate"])
        train_data, valid_data = random_split(
            train_data,
            [len(train_data) - n_valid, n_valid],
            generator=torch.Generator().manual_seed(config["valid"]["seed"]),
        )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=config["train"]["learning_rate"], momentum=0.9
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["train"]["epochs"] - 1
    )

    print("train start ...")
    steps = 1
    model.train()
    for epoch in range(config["train"]["epochs"]):
        for images, labels in tqdm(
            DataLoader(
                dataset=train_data, batch_size=config["train"]["batch"], shuffle=True
            ),
            desc=f"Progress {epoch + 1} epoch",
        ):
            loss = loss_fn(model(images), labels.to(config["device"]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if config["train"]["log"] and steps % config["train"]["log_steps"] == 0:
                print(
                    f"epochs: {epoch + 1},\tsteps: {steps},\tloss:{round(loss.item(), 8)}"
                )

            if config["valid"]["do_valid"] and steps % config["valid"]["steps"] == 0:
                evaluation(model, valid_data, config, valid=True)
                model.train()

            steps += 1

        evaluation(model, valid_data, config, valid=True)
        model.train()
        scheduler.step()
