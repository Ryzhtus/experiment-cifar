import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

class CIFAR10(LightningModule):
    def __init__(self, model, config, train, eval, test) -> None:
        super().__init__()

        self.model = model
        self.config = config
        
        self.train_data = train
        self.eval_data = eval
        self.test_data = test

        self.loss_fn = nn.CrossEntropyLoss()

        self.correct = 0
        self.total = 0

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.config["lr"], 
                              momentum=self.config["momentum"])
        
        return {"optimizer": optimizer}
    
    def process_batch(self, batch, stage=None):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        self.total += labels.size(0)
        self.correct += (predicted == labels).sum().item()

        if stage:
            self.log(f"{stage}_loss", loss.item(), prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch, "valid")

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch, "test")

        return loss

    def training_epoch_end(self, _):
        self.log(f"train_acc", 100 * self.correct // self.total, prog_bar=True)
        self.correct = 0
        self.total = 0

    def validation_epoch_end(self, _):
        self.log(f"valid_acc", 100 * self.correct // self.total, prog_bar=True)
        self.correct = 0
        self.total = 0

    def testing_epoch_end(self, _):
        self.log(f"test_acc", 100 * self.correct // self.total, prog_bar=True)
        self.correct = 0
        self.total = 0

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.config["batch_size"], shuffle=True)
    
    def validation_dataloader(self):
        return DataLoader(self.eval_data, batch_size=self.config["batch_size"], shuffle=False)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.config["batch_size"], shuffle=False)