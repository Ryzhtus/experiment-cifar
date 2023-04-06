import torch
import wandb
from datetime import datetime

def train(model, train_dataloader, loss_fn, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'Train: [{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            wandb.log({"Train": epoch + 1, "Step": f"{i + 1:5d}", "train_loss": f"{running_loss / 2000:.3f}"})
            running_loss = 0.0

def eval(model, eval_dataloader, loss_fn, epoch):
    model.eval()
    running_loss = 0.0
    for i, data in enumerate(eval_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        with torch.no_grad():
            # forward + backward + optimize
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 2000 mini-batches
            print(f'Valid: [{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            wandb.log({"Valid": epoch + 1, "Step": f"{i + 1:5d}", "val_loss": f"{running_loss / 2000:.3f}"})
            running_loss = 0.0

def test(model, test_dataloader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
    wandb.log({"Test accuracy": 100 * correct // total})

def run(model, train_dataloader, eval_dataloader, test_dataloader, loss_fn, optimizer, epochs):
    
    print("Started Training")
    for epoch in range(epochs):
        train(model, train_dataloader, loss_fn, optimizer, epoch)
        eval(model, eval_dataloader, loss_fn, epoch)

    print('Finished Training')
    print("Saving the checkpoint")
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y-%H%M")
    PATH = f'checkpoints/cifar-model-{dt_string}.pth'
    torch.save(model.state_dict(), PATH)
    print("Testing")

    test(model, test_dataloader)

    print("Finished")

