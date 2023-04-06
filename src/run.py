import torch.nn as nn
import torch.optim as optim
from dataset import CIFAR10_Dataset
from model import Model
from utils import read_cifar, read_cifar_meta, set_seed
from train import run
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from omegaconf import OmegaConf
import wandb

wandb.init(project="cifar10", dir="logs")

config = OmegaConf.load("configs/model2.yaml")

train, train_targets = read_cifar()
test, test_targets = read_cifar(False)
classes, class_to_idx = read_cifar_meta()

train, eval, train_targets, eval_targets = train_test_split(train, train_targets, test_size=0.2, random_state=config["seed"])

train_dataset = CIFAR10_Dataset(train, train_targets)
train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

eval_dataset = CIFAR10_Dataset(eval, eval_targets)
eval_dataloader = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=False)

test_dataset = CIFAR10_Dataset(test, test_targets)
test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

model = Model()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

run(model, train_dataloader, eval_dataloader, test_dataloader, loss_fn, optimizer, config["epochs"])