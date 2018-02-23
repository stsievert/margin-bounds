from __future__ import print_function
import argparse
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import numpy.linalg as LA

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--dataset', type=str, default='MNIST',
                    help='Which dataset to use?')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.dataset not in ['MNIST', 'FashionMNIST']:
    raise ValueError("`dataset` not recognized")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
dataset = getattr(datasets, args.dataset)
train_loader = torch.utils.data.DataLoader(
    dataset(f'data/{args.dataset}', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    dataset(f'data/{args.dataset}', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()
if args.cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer.steps = 0
max_iter = args.epochs * 50e3 / args.batch_size
_inspect = np.logspace(np.log10(4), np.log10(max_iter), num=20, dtype=int)
_inspect = list(_inspect.astype(int))
print(_inspect)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        optimizer.steps += 1
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                optimizer.steps, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
        if optimizer.steps in _inspect:
            break

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        #  margin = output.data.max

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    n = len(test_loader.dataset)
    return {'accuracy': correct / n, 'loss': test_loss}


def _get_weights(model):
    W = [getattr(model, name) for name in ['conv1', 'conv2', 'fc1', 'fc2']]
    weights = [w.weight.view(w.weight.size()[0], -1) for w in W]
    matrices = [w.cat((w, W.bias), dim=-1) for w, W in zip(weights, W)]
    return matrices


def stats(model, loader, X_fro_norm):
    global diff, row, i, idx
    margins = torch.Tensor()
    for data, target in loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        diff = output.data.max(1, keepdim=True)[0] - output.data

        diff = diff[diff > 0].view(diff.size()[0], diff.size()[1] - 1)
        margin = diff.min(1)[0]
        margins = torch.cat((margins, margin))
    # S = prod(max(sigma(A)) * L_i)
    # T = sum_i ||A_i - M_i]^{2/3}_1 / || ||A_i||^{2/3}_2
    # R = T**(3/2) * S
    # Lipschitz constants: relu: 1. max_pooling lipschitz: 1. log-softmax: 1
    # TODO: calculate margin with *all* data points
    A = _get_weights(model)
    L2norms = [LA.norm(a.data.numpy(), ord=2) for a in A]
    L1norms = [LA.norm(a.data.numpy().flat[:], ord=1) for a in A]
    T = sum(l1**(2/3) / l2**(2/3) for l1, l2 in zip(L1norms, L2norms))
    S = np.prod(L2norms)
    R = T**(3/2) * S
    n = len(loader.dataset)
    print(margins.size(), R, X_fro_norm, n)
    margin_dist = margins / (R * X_fro_norm / n)

    return {'margin_dist': margin_dist}


if __name__ == "__main__":
    data = []
    X = torch.cat(input for input, _ in train_loader)
    X = X.view(X.size()[0], -1)
    sq_terms = (X.numpy().flat[:]**2).sum()
    X_fro_norm = np.sqrt(sq_terms)
    for epoch in range(1, args.epochs + 1):
        print("epoch =", epoch)
        train(epoch)
        datum = {'steps': optimizer.steps,
                 'epochs': optimizer.steps / len(train_loader.dataset),
                 **stats(model, train_loader, X_fro_norm), **test()}
        data += [datum]
        with open(f'./sims-{args.dataset}.pkl', 'wb') as f:
            pickle.dump(data, f)
