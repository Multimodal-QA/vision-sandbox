from torchvision import datasets, transforms


class CifarDataLoader:
    def __init__(self, data_config: dict):
        self.data_dir = data_config["root_dir"]
        self.img_size = data_config["img_size"]

        norm = {
            "mean": tuple(map(float, data_config["norm"]["mean"].split())),
            "stdev": tuple(map(float, data_config["norm"]["stdev"].split())),
        }

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((data_config["img_size"], data_config["img_size"])),
                transforms.Normalize(norm["mean"], norm["stdev"]),
            ]
        )

    def load_train_data(self):
        return datasets.CIFAR100(
            root=self.data_dir, train=True, download=True, transform=self.transform
        )

    def load_valid_data(self):
        pass

    def load_test_data(self):
        return datasets.CIFAR100(
            root=self.data_dir, train=False, download=True, transform=self.transform
        )
