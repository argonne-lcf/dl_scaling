# Pytorch DDP example for mnist

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time

# 1. Import necessary packages.
import os
import socket
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 2. Set global variables for rank, local_rank, world size
try:
    from mpi4py import MPI

    with_ddp=True
    #local_rank = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    size = MPI.COMM_WORLD.Get_size() # dist.get_world_size()?
    rank = MPI.COMM_WORLD.Get_rank() # dist.get_rank()?
    local_rank = rank%4

    # Pytorch will look for these:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)

    # It will want the master address too, which we'll broadcast:
    if rank == 0:
        master_addr = socket.gethostname()
    else:
        master_addr = None

    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(2345)
    print("DDP: I am worker %s of %s. My local rank is %s" %(rank, size, local_rank))
    # MPI.COMM_WORLD.Barrier()

except Exception as e:
    with_ddp=False
    local_rank = 0
    size = 1
    rank = 0
    print("MPI initialization failed!")
    print(e)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def metric_average(val, name):
    if (with_ddp):
        # Sum everything and divide by total size:
        dist.all_reduce(val,op=dist.reduce_op.SUM)
        val /= size
    else:
        pass
    return val

def train(args, model, device, train_loader, optimizer, epoch, train_sampler):
    model.train()
    # 8.DDP: set epoch to sampler for shuffling.
    running_loss = torch.tensor(0.0)
    training_acc = torch.tensor(0.0)
    running_loss = running_loss.cuda()
    training_acc = training_acc.cuda()
    train_sampler.set_epoch(epoch)
    if rank == 0:
        print("Starting training")
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        training_acc += pred.eq(target.data.view_as(pred)).float().sum()
        running_loss += loss
        if batch_idx % args.log_interval == 0 and rank == 0:
            print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(rank,
                epoch, batch_idx * len(data), len(train_sampler), 100. * batch_idx / len(train_loader), loss.item()/args.batch_size))
    
            if args.dry_run:
                break

    running_loss /= len(train_sampler)
    training_acc /= len(train_sampler)
    loss_avg = metric_average(running_loss, 'running_loss')
    training_acc = metric_average(training_acc, 'training_acc')
    if rank==0: print("Training set: Average loss: {:.4f}, Accuracy: {:.2f}%".format(loss_avg, training_acc*100))

def test(model, device, test_loader, test_sampler):
    model.eval()
    test_loss = 0
    correct = 0
    test_loss = torch.tensor(0.0)
    correct = torch.tensor(0.0)
    test_loss = test_loss.cuda()
    correct = correct.cuda()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_sampler)
    correct /= len(test_sampler)

    test_loss = metric_average(test_loss, 'avg_loss')
    correct = metric_average(correct, 'avg_accuracy')

    if rank==0:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    # if rank == 0:
    #     print('Test set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
    #         test_loss, 100. * test_accuracy))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    # 3.DDP: Initialize library.
    if with_ddp:
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')

    torch.manual_seed(args.seed)

    # 4.DDP: Pin GPU to local rank.
    torch.cuda.set_device(int(local_rank))

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    # 5.DDP: Use DistributedSampler to partition the training data.
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset1, num_replicas=size, rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset1, sampler=train_sampler, **train_kwargs)
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset2, num_replicas=size, rank=rank)
    test_loader = torch.utils.data.DataLoader(dataset2, sampler=test_sampler, **test_kwargs)

    # model = Net().to(device)
    model = Net()  
    model = model.cuda()
    # 6.Wrap the model in DDP:
    if with_ddp:
        model = DDP(model)

    # 7.DDP: scale learning rate by the number of GPUs.
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr * size)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, train_sampler)
        test(model, device, test_loader, test_sampler)
        scheduler.step()

    if rank==0:
        print(time.time()-t0)
        if args.save_model:
            torch.save(model.state_dict(), "mnist_ddp.pt")


if __name__ == '__main__':
    main()
