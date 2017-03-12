#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import torchbiomed.datasets as dset
import torchbiomed.transforms as biotransforms
import torchbiomed.loss as bioloss

import os
import sys
import math

import shutil

import setproctitle

import vnet
import make_graph

nodule_masks="luna16_nodule_masks"
lung_masks="luna16_seg_lungs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=1)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/vnet.base'
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("build vnet")
    net = vnet.VNet(args.batchSz, inplace=False)

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        print("moving network to GPU")
        net = net.cuda()

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    # trainTransform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    # testTransform = transforms.Compose([
    #     transforms.ToTensor(),
    # ])

    trainTransform = None
    testTransform = None
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    print("loading training set")
    trainLoader = DataLoader(
        dset.LUNA16(root='luna16', images="luna16_ct_normalized", targets=lung_masks,
                    train=True, transform=trainTransform),
        batch_size=args.batchSz, shuffle=True, **kwargs)
    print("loading test set")
    testLoader = DataLoader(
        dset.LUNA16(root='luna16', images="luna16_ct_normalized", targets=lung_masks,
                    train=False, transform=testTransform),
        batch_size=args.batchSz, shuffle=False, **kwargs)

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                            momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=1e-4)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF)
        test(args, epoch, net, testLoader, optimizer, testF)
        torch.save(net, os.path.join(args.save, 'latest.pth'))
        os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()

def train(args, epoch, net, trainLoader, optimizer, trainF):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        loss = bioloss.dice_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        err = 100.*(1. - loss.data[0])
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.8f}\tError: {:.8f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()

def test(args, epoch, net, testLoader, optimizer, testF):
    net.eval()
    test_loss = 0
    incorrect = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = net(data)
        loss = bioloss.dice_loss(output, target).data[0]
        test_loss += loss
        incorrect += (1. - loss)

    test_loss /= len(testLoader) # loss function already averages over batch size
    nTotal = len(testLoader.dataset)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()

def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150: lr = 1e-1
        elif epoch == 150: lr = 1e-2
        elif epoch == 225: lr = 1e-3
        else: return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__=='__main__':
    main()
