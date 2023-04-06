from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from dataset import CIFAR10_Dataset
from model import Model
from utils import read_cifar, read_cifar_meta
from sklearn.model_selection import train_test_split
from pl_trainer import CIFAR10
from omegaconf import OmegaConf

config = OmegaConf.load("configs/model2.yaml")

train, train_targets = read_cifar()
test, test_targets = read_cifar(False)
classes, class_to_idx = read_cifar_meta()

train, eval, train_targets, eval_targets = train_test_split(train, train_targets, test_size=0.2, random_state=config["seed"])

train_dataset = CIFAR10_Dataset(train, train_targets)
eval_dataset = CIFAR10_Dataset(eval, eval_targets)
test_dataset = CIFAR10_Dataset(test, test_targets)

model = Model()

wandb_logger = WandbLogger(project="cifar10", dir="logs")
wandb_logger.log_hyperparams(config)

lightning_model = CIFAR10(model, config, train_dataset, eval_dataset, test_dataset)

trainer = Trainer(logger=wandb_logger)

trainer.fit(lightning_model)
trainer.test(lightning_model)

