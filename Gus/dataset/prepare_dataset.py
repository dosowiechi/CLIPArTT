import math
import torch
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torchvision.datasets import ImageFolder
from dataset.visdatest import *

NORM = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
te_transforms = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(*NORM)])
tr_transforms = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(*NORM)])

visda_train = transforms.Compose([transforms.Resize((256,256)),
                                  transforms.RandomCrop((224,224)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

visda_val = transforms.Compose([transforms.Resize((256,256)),
                                transforms.CenterCrop((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

#augment_transforms = transforms.Compose([transforms.RandomRotation(180),
#                                         transforms.ColorJitter()])

augment_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

office_test = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])




common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                      'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                      'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

def prepare_test_data(args):
    if args.dataset == 'cifar10':
        tesize = 10000
        if not hasattr(args, 'corruption') or args.corruption == 'original':
            teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                train=False, download=False, transform=te_transforms)
        elif args.corruption in common_corruptions:
            teset_raw = np.load(args.dataroot + '/CIFAR-10-C/%s.npy' % (args.corruption))
            teset_raw = teset_raw[(args.level - 1) * tesize: args.level * tesize]
            teset = torchvision.datasets.CIFAR10(root=args.dataroot,
                train=False, download=False, transform=te_transforms)
            teset.data = teset_raw

        elif args.corruption == 'cifar_new':
            from utils.cifar_new import CIFAR_New
            teset = CIFAR_New(root=args.dataroot + '/CIFAR10.1', transform=te_transforms)
            permute = False
        else:
            raise Exception('Corruption not found!')

    elif args.dataset == 'cifar100':
        tesize = 10000
        if not hasattr(args, 'corruption') or args.corruption == 'original':
            teset = torchvision.datasets.CIFAR100(root=args.dataroot  + 'test/',
                train=False, download=False, transform=te_transforms)
        elif args.corruption in common_corruptions:
            teset_raw = np.load(args.dataroot + '/CIFAR-100-C/%s.npy' % (args.corruption))
            teset_raw = teset_raw[(args.level - 1) * tesize: args.level * tesize]
            teset = torchvision.datasets.CIFAR100(root=args.dataroot,
                train=False, download=True, transform=te_transforms)

            teset.data = teset_raw
    elif args.dataset == 'visda':
        teset = VisdaTest(args.dataroot, transforms=visda_val)
    elif args.dataset == 'office':
        teset = ImageFolder(root=args.dataroot + 'OfficeHomeDataset_10072016/' + args.category, transform=office_test)
    else:
        raise Exception('Dataset not found!')

    if args.distributed:
        te_sampler = torch.utils.data.distributed.DistributedSampler(teset)
    else:
        te_sampler = None

    if not hasattr(args, 'workers'):
        args.workers = 1
    if args.distributed:
        teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
            shuffle=(te_sampler is None), num_workers=args.workers, pin_memory=True, sampler=te_sampler, drop_last=True)
    else:
        teloader = torch.utils.data.DataLoader(teset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)

    return teloader, te_sampler


def prepare_val_data(args, transform=None):
    if args.dataset == 'visda':
        vset = ImageFolder(root=args.dataroot + 'validation/', transform=transform if transform is not None else visda_val)
    else:
        raise Exception('Dataset not found!')

    if args.distributed:
        v_sampler = torch.utils.data.distributed.DistributedSampler(vset)
    else:
        v_sampler = None
    if not hasattr(args, 'workers'):
        args.workers = 1
    vloader = torch.utils.data.DataLoader(vset, batch_size=args.batch_size,
        shuffle=(v_sampler is None), num_workers=args.workers, pin_memory=True, sampler=v_sampler, drop_last=True)
    return vloader, v_sampler

def prepare_train_data(args, transform=None):
    if args.dataset == 'cifar10':
        trset = torchvision.datasets.CIFAR10(root=args.dataroot,
            train=True, download=False, transform=tr_transforms if transform is None else transform)
        vset = None
    elif args.dataset == 'cifar100':
        trset = torchvision.datasets.CIFAR100(root=args.dataroot + 'train/',
            train=True, download=False, transform=tr_transforms if transform is None else transform)
        vset = None
    elif args.dataset == 'visda':
        dataset = ImageFolder(root=args.dataroot + 'train/', transform=visda_train if transform is None else transform)
        trset, vset = random_split(dataset, [106678, 45719], generator=torch.Generator().manual_seed(args.seed))
    elif args.dataset == 'office':
        dataset = ImageFolder(root=args.dataroot + 'OfficeHomeDataset_10072016/' + args.category, transform=augment_transforms if transform is None else transform)
        long = len(dataset)
        trset, vset = random_split(dataset, [math.floor(long*0.8), math.floor(long*0.2)], generator=torch.Generator().manual_seed(args.seed))
    else:
        raise Exception('Dataset not found!')

    if args.distributed:
        tr_sampler = torch.utils.data.distributed.DistributedSampler(trset)
        if args.dataset == 'visda':
            v_sampler = torch.utils.data.distributed.DistributedSampler(vset)
        else:
            v_sampler = None
    else:
        tr_sampler = None
        v_sampler = None

    if not hasattr(args, 'workers'):
        args.workers = 1
    trloader = torch.utils.data.DataLoader(trset, batch_size=args.batch_size,
        shuffle=(tr_sampler is None), num_workers=args.workers, pin_memory=True, sampler=tr_sampler, drop_last=True)
    if args.dataset == 'visda':
        vloader = torch.utils.data.DataLoader(vset, batch_size=args.batch_size,
            shuffle=(v_sampler is None), num_workers=args.workers, pin_memory=True, sampler=v_sampler, drop_last=True)
    else:
        vloader = None
    return trloader, tr_sampler, vloader, v_sampler