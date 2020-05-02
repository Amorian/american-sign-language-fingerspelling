from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device('cuda')

parser = argparse.ArgumentParser(description='ASL Finger Spelling')
parser.add_argument('--data', type=str, default='asl_dataset', metavar='D',
                    help="default: data")
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='default: 64')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='default: 30')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='default: 0.01')
parser.add_argument('--decay', type=float, default=1e-6, metavar='DY',
                    help='default: 1e-6')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='default: 0.9')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='default: 10')
args = parser.parse_args()

torch.manual_seed(args.seed)

train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/train', transforms.Compose([
                        transforms.Resize((28, 28)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])),
    batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(args.data + '/test', transforms.Compose([
                        transforms.Resize((28, 28)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])),
    batch_size=args.batch_size, shuffle=False, num_workers=0)


from model import Net
model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.decay,
                      momentum=args.momentum, nesterov=True)

train_loss_epoch = []
validation_loss_epoch = []
train_acc_epoch = []
validation_acc_epoch = []

def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        sum_train_loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        optimizer.step()
        train_loss += sum_train_loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_loss /= len(train_loader.dataset)
    train_loss_epoch.append(train_loss)
    print('\nTraining set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    train_acc_epoch.append(100. * correct / len(train_loader.dataset))
def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        validation_loss += F.nll_loss(output, target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    validation_loss /= len(val_loader.dataset)
    validation_loss_epoch.append(validation_loss)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    validation_acc_epoch.append(100. * correct / len(train_loader.dataset))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    validation()
    model_file = 'model_' + str(epoch) + '.pth'
    torch.save(model.state_dict(), model_file)
    print(model_file)

plt.figure()
plt.plot(range(1, args.epochs + 1), train_loss_epoch, label = 'Training Loss')
plt.plot(range(1, args.epochs + 1), validation_loss_epoch, label = 'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Epochs vs Loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(range(1, args.epochs + 1), train_acc_epoch)
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.title('Epochs vs Accuracy')
plt.show()

plt.figure()
plt.plot(range(1, args.epochs + 1), validation_acc_epoch)
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.title('Epochs vs Accuracy')
plt.show()
