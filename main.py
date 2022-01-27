import torch
import yaml

from constants import CONFIG_DIR
from evaluation import evaluation
from model import ClsModel
from train import train
from utils.load_data import CifarDataLoader

# TODO: Add logging


def main():
    with open(CONFIG_DIR + "/config.yaml") as file:
        config = yaml.safe_load(file)

    config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Load data ...")
    loader = CifarDataLoader(config["data"])
    train_data = loader.load_train_data()
    if config["eval"]["do_eval"]:
        test_data = loader.load_test_data()

    print("initialize model ...")
    model = ClsModel(
        config["model"]["name"], config["device"], img_size=config["data"]["img_size"]
    )

    train(model, train_data, config)

    if config["eval"]["do_eval"]:
        evaluation(model, test_data, config)


if __name__ == "__main__":
    main()
