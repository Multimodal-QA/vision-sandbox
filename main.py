from typing import Optional

import timm
import torch
import yaml
from tqdm import tqdm
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from constants import CONFIG_DIR

# TODO: Add logging


class ClsModel(nn.Module):
    def __init__(self, name: str, device: torch.device,
                 pretrained: Optional[bool] = True,
                 img_size: Optional[int] = 224,
                 num_class: Optional[int] = 100):
        super(ClsModel, self).__init__()
        self.model = timm.create_model(name, pretrained=pretrained, img_size=img_size, num_classes=num_class)
        self.device = device
        self.model.to(device)
        self.score_fn = nn.Softmax(dim=1)

    def forward(self, x):
        return self.model(x.to(self.device))

    def predict(self, x):
        scores = self.score_fn(self.model(x.to(self.device)))
        return torch.argmax(scores, dim=1)


def evaluation(model: ClsModel, data, config, valid: Optional[bool] = False):
    model.eval()

    correct = torch.tensor(0, dtype=torch.int, device=config['device'])
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset=data, batch_size=config['valid' if valid else 'eval']['batch']),
                                   desc='Validation' if valid else 'Evaluation'):
            preds = model.predict(images)
            correct += torch.sum(preds == labels.to(config['device']))

    print(f'top1 acc: {round(correct.item() / len(data), 8)}')


def train(model: ClsModel, train_data, loss_fn, config):
    if config['valid']['do_valid']:
        print('validation will be proceeded during training')
        assert config['valid']['steps'] > 0

    if config['valid']['train_split']:
        print('split data ...')
        assert config['valid']['rate'] > 0
        n_valid = int(len(train_data) * config['valid']['rate'])
        train_data, valid_data = random_split(train_data, [len(train_data) - n_valid, n_valid],
                                              generator=torch.Generator().manual_seed(config['valid']['seed']))

    # TODO: Modify hard code for optimizer
    optimizer = optim.SGD(model.parameters(), lr=config['train']['learning_rate'], momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'])

    print('train start ...')
    steps = 1
    model.train()
    for epoch in range(config['train']['epochs']):
        for images, labels in tqdm(DataLoader(dataset=train_data, batch_size=config['train']['batch'], shuffle=True),
                                   desc=f'Progress {epoch + 1} epoch'):
            loss = loss_fn(model(images), labels.to(config['device']))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if config['train']['log'] and steps % config['train']['log_steps'] == 0:
                print(f'epochs: {epoch + 1},\tsteps: {steps},\tloss:{round(loss.item(), 8)}')

            if config['valid']['do_valid'] and steps % config['valid']['steps'] == 0:
                evaluation(model, valid_data, config, valid=True)
                model.train()

            steps += 1

        scheduler.step()


def main():
    with open(CONFIG_DIR + '/config.yaml') as file:
        config = yaml.safe_load(file)

    data_dir = config['data']['root_dir']
    img_size = config['data']['img_size']
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    norm = {
        'mean': tuple(map(float, config['data']['norm']['mean'].split())),
        'stdev': tuple(map(float, config['data']['norm']['stdev'].split()))
    }

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(norm['mean'], norm['stdev'])
    ])

    print('Load data ...')  # TODO: Modify hard code for dataset
    train_data = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)

    print('initialize model ...')
    model = ClsModel(config['model']['name'], config['device'], img_size=img_size)

    loss_fn = nn.CrossEntropyLoss()

    train(model, train_data, loss_fn, config)

    if config['eval']['do_eval']:
        test_data = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
        evaluation(model, test_data, config)


if __name__ == '__main__':
    main()
