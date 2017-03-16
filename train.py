#!/usr/bin/env python3

from local import *
import time
import argparse
import torch

import torch.nn as nn
import torch.nn.init as init
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

nodule_masks = "luna16_nodule_masks"
lung_masks = "luna16_seg_lungs"


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        nn.init.kaiming_normal(m.weight)
    elif classname.find('BatchNorm') != -1:
        #nn.init.kaiming_normal(m.weight)
        m.bias.data.fill_(0)

def do_disable(m):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        m.training = False


def datestr():
    now = time.gmtime()
    return '{}{:02}{:02}_{:02}{:02}'.format(now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSz', type=int, default=10)
    parser.add_argument('--nll', type=bool, default=True)
    parser.add_argument('--PReLU', type=bool, default=True)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--nEpochs', type=int, default=300)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='sgd',
                        choices=('sgd', 'adam', 'rmsprop'))
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.save = args.save or 'work/vnet.base.{}'.format(datestr())
    nll = args.nll
    setproctitle.setproctitle(args.save)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    print("build vnet")
    net = vnet.VNet(elu=False, nll=nll)
    batch_size = args.batchSz
    if args.ngpu > 1:
        gpu_ids = range(args.ngpu)
        net = nn.parallel.DataParallel(net, device_ids=gpu_ids)
        batch_size = args.ngpu*args.batchSz

    net.apply(weights_init)
    weight_decay = 1e-4

    if nll:
        train = train_nll
        test = test_nll
        class_balance = True
    else:
        train = train_dice
        test = test_dice
        class_balance = False

    print('  + Number of params: {}'.format(
        sum([p.data.nelement() for p in net.parameters()])))
    if args.cuda:
        net = net.cuda()

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    normMu = [-642.794]
    normSigma = [459.512]
    normTransform = transforms.Normalize(normMu, normSigma)

    trainTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    print("loading training set")
    trainSet = dset.LUNA16(root='luna16', images="luna16_ct_normalized", targets=lung_masks,
                           train=True, transform=trainTransform, allow_empty=False,
                           class_balance=class_balance, split=[2,2,2], seed=args.seed)
    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, **kwargs)
    print("loading test set")
    testLoader = DataLoader(
        dset.LUNA16(root='luna16', images="luna16_ct_normalized", targets=lung_masks,
                    train=False, transform=testTransform, allow_empty=False, seed=args.seed,
                    split=[2, 2, 2]),
        batch_size=batch_size, shuffle=False, **kwargs)

    target_weight = trainSet.target_weight()
    print(target_weight)
    bg_weight = 1.0 - target_weight
    class_weights = torch.FloatTensor([bg_weight, target_weight])
    if args.cuda:
        class_weights = class_weights.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,
                              momentum=0.99, weight_decay=weight_decay)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), weight_decay=weight_decay)

    trainF = open(os.path.join(args.save, 'train.csv'), 'w')
    testF = open(os.path.join(args.save, 'test.csv'), 'w')

    for epoch in range(1, args.nEpochs + 1):
        adjust_opt(args.opt, optimizer, epoch)
        train(args, epoch, net, trainLoader, optimizer, trainF, class_weights)
        test(args, epoch, net, testLoader, optimizer, testF, class_weights)
        torch.save(net, os.path.join(args.save, 'latest.pth'))
        os.system('./plot.py {} &'.format(args.save))

    trainF.close()
    testF.close()


def train_nll(args, epoch, net, trainLoader, optimizer, trainF, weights):
    net.train()
    nProcessed = 0
    nTrain = len(trainLoader.dataset)
    for batch_idx, (data, target) in enumerate(trainLoader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = net(data)
        target = target.view(target.numel())
        loss = F.nll_loss(output, target, weight=weights)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/target.numel()
        partialEpoch = epoch + batch_idx / len(trainLoader) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. * batch_idx / len(trainLoader),
            loss.data[0], err))

        trainF.write('{},{},{}\n'.format(partialEpoch, loss.data[0], err))
        trainF.flush()


def test_nll(args, epoch, net, testLoader, optimizer, testF, weights):
    # we don't want to actually call net.eval() as we need
    # to maintain the running stats to get good results at
    # test time
    net.apply(do_disable)
    test_loss = 0
    incorrect = 0
    numel = 0
    for data, target in testLoader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        target = target.view(target.numel())
        numel += target.numel()
        output = net(data)
        test_loss += F.nll_loss(output, target, weight=weights).data[0]
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        incorrect += pred.ne(target.data).cpu().sum()

    test_loss /= len(testLoader)  # loss function already averages over batch size
    err = 100.*incorrect/numel
    print('\nTest set: Average loss: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, numel, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()


def train_dice(args, epoch, net, trainLoader, optimizer, trainF, weights):
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


def test_dice(args, epoch, net, testLoader, optimizer, testF, weights):
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

    test_loss /= len(testLoader)  # loss function already averages over batch size
    nTotal = len(testLoader)
    err = 100.*incorrect/nTotal
    print('\nTest set: Average Dice Coeff: {:.4f}, Error: {}/{} ({:.0f}%)\n'.format(
        test_loss, incorrect, nTotal, err))

    testF.write('{},{},{}\n'.format(epoch, test_loss, err))
    testF.flush()


def adjust_opt(optAlg, optimizer, epoch):
    if optAlg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

if __name__ == '__main__':
    main()
