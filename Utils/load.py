import torch
import numpy as np
import random
import os
from torchvision import datasets, transforms
import torch.optim as optim
from Models import mlp
from Models import lottery_vgg
from Models import lottery_resnet
from Models import tinyimagenet_vgg
from Models import tinyimagenet_resnet
from Models import imagenet_vgg
from Models import imagenet_resnet
from Pruners import pruners
from Utils import custom_datasets

def device(gpu):
    use_cuda = torch.cuda.is_available()
    return torch.device(("cuda:" + str(gpu)) if use_cuda else "cpu")

def dimension(dataset):
    if dataset == 'mnist':
        input_shape, num_classes = (1, 28, 28), 10
    if dataset == 'cifar10':
        input_shape, num_classes = (3, 32, 32), 10
    if dataset == 'cifar100':
        input_shape, num_classes = (3, 32, 32), 100
    if dataset == 'tiny-imagenet':
        input_shape, num_classes = (3, 64, 64), 200
    if dataset == 'imagenet':
        input_shape, num_classes = (3, 224, 224), 1000
    return input_shape, num_classes

def get_transform(size, padding, mean, std, preprocess):
    transform = []
    if preprocess:
        transform.append(transforms.RandomCrop(size=size, padding=padding))
        transform.append(transforms.RandomHorizontalFlip())
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize(mean, std))
    return transforms.Compose(transform)

def dataloader(dataset_name, batch_size, prune_batch_size, train, workers, seed, length, root=None):
    # Dataset
    if dataset_name == 'mnist':
        mean, std = (0.1307,), (0.3081,)
        transform = get_transform(size=28, padding=0, mean=mean, std=std, preprocess=False)
        datadir = root
        dataset = datasets.MNIST(datadir, train=train, download=True, transform=transform)
    if dataset_name == 'cifar10':
        mean, std = (0.491, 0.482, 0.447), (0.247, 0.243, 0.262)
        transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
        datadir = root
        dataset = datasets.CIFAR10(datadir, train=train, download=True, transform=transform) 
    if dataset_name == 'cifar100':
        mean, std = (0.507, 0.487, 0.441), (0.267, 0.256, 0.276)
        transform = get_transform(size=32, padding=4, mean=mean, std=std, preprocess=train)
        datadir = root
        dataset = datasets.CIFAR100(datadir, train=train, download=True, transform=transform)
    if dataset_name == 'tiny-imagenet':
        mean, std = (0.480, 0.448, 0.397), (0.276, 0.269, 0.282)
        transform = get_transform(size=64, padding=4, mean=mean, std=std, preprocess=train)
        datadir = root
        dataset = custom_datasets.TINYIMAGENET(datadir, train=train, download=False, transform=transform)
    if dataset_name == 'imagenet':
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        if train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.2,1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        datadir = root
        if train:
            folder = f'{datadir}/train'
        else:
            folder = f'{datadir}/val'
        dataset = datasets.ImageFolder(folder, transform=transform)
    
    # Dataloader
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    # shuffle = train is True

    if train:
        if not dataset_name == 'imagenet':
            indices       = torch.randperm(len(dataset))
            train_size    = int(0.8 * len(dataset))
            train_dataset = torch.utils.data.Subset(dataset, indices[:train_size])
            val_dataset   = torch.utils.data.Subset(dataset, indices[train_size:])

            indices_prune = torch.randperm(len(train_dataset))
            prune_dataset = torch.utils.data.Subset(train_dataset, indices_prune[:length])

            g = torch.Generator()
            g.manual_seed(seed)

            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                worker_init_fn=seed_worker,
                generator=g,
                **kwargs
                )

            val_loader = torch.utils.data.DataLoader(
                dataset=val_dataset, 
                batch_size=batch_size, 
                shuffle=False, 
                worker_init_fn=seed_worker,
                generator=g,
                **kwargs
                )
            
            prune_loader = torch.utils.data.DataLoader(
                dataset=prune_dataset, 
                batch_size=prune_batch_size, 
                shuffle=True, 
                worker_init_fn=seed_worker,
                generator=g,
                **kwargs
                )
            print(f'---------- DATASET : {dataset_name} ----------')
            print(f'Train Loader size : {len(train_loader.dataset)}')
            print(f'Val Loader size :   {len(val_loader.dataset)}')
            print(f'Prune Loader size : {len(prune_loader.dataset)}')
        else:
            indices       = torch.randperm(len(dataset))
            train_dataset = dataset
            indices_prune = torch.randperm(len(train_dataset))
            prune_dataset = torch.utils.data.Subset(train_dataset, indices_prune[:length])

            g = torch.Generator()
            g.manual_seed(seed)

            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset, 
                batch_size=batch_size, 
                shuffle=True, 
                worker_init_fn=seed_worker,
                generator=g,
                **kwargs
                )

            val_loader = None
            
            prune_loader = torch.utils.data.DataLoader(
                dataset=prune_dataset, 
                batch_size=prune_batch_size, 
                shuffle=True, 
                worker_init_fn=seed_worker,
                generator=g,
                **kwargs
                )
            print(f'---------- DATASET : {dataset_name} ----------')
            print(f'Train Loader size : {len(train_loader.dataset)}')
            print(f'Val Loader :        None')
            print(f'Prune Loader size : {len(prune_loader.dataset)}')
        return train_loader, val_loader, prune_loader
    else:
        g = torch.Generator()
        g.manual_seed(seed)

        test_loader = torch.utils.data.DataLoader(
            dataset=dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            worker_init_fn=seed_worker,
            generator=g,
            **kwargs
            )

        val_loader   = None
        prune_loader = None

        print(f'---------- DATASET : {dataset_name} ----------')
        print(f'Test Loader size : {len(test_loader.dataset)}')

    return test_loader, val_loader, prune_loader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_dataloaders(dataset_name, batch_size, prune_batch_size, workers, seed, length, root=None):

    train_loader, val_loader, prune_loader = dataloader(
        dataset_name=dataset_name, batch_size=batch_size, prune_batch_size=prune_batch_size, train=True, workers=workers, seed=seed, length=length, root=root)
    
    test_loader, _, _ = dataloader(
        dataset_name=dataset_name, batch_size=batch_size, prune_batch_size=prune_batch_size, train=False, workers=workers, seed=seed, length=length, root=root)
    
    return train_loader, val_loader, test_loader, prune_loader




def model(model_architecture, model_class):
    default_models = {
        'fc' : mlp.fc,
        'conv' : mlp.conv,
    }
    lottery_models = {
        'vgg11' : lottery_vgg.vgg11,
        'vgg11-bn' : lottery_vgg.vgg11_bn,
        'vgg13' : lottery_vgg.vgg13,
        'vgg13-bn' : lottery_vgg.vgg13_bn,
        'vgg16' : lottery_vgg.vgg16,
        'vgg16-bn' : lottery_vgg.vgg16_bn,
        'vgg19' : lottery_vgg.vgg19,
        'vgg19-bn' : lottery_vgg.vgg19_bn,
        'resnet20': lottery_resnet.resnet20,
        'resnet32': lottery_resnet.resnet32,
        'resnet44': lottery_resnet.resnet44,
        'resnet56': lottery_resnet.resnet56,
        'resnet110': lottery_resnet.resnet110,
        'resnet1202': lottery_resnet.resnet1202,
        'wide-resnet20': lottery_resnet.wide_resnet20,
        'wide-resnet32': lottery_resnet.wide_resnet32,
        'wide-resnet44': lottery_resnet.wide_resnet44,
        'wide-resnet56': lottery_resnet.wide_resnet56,
        'wide-resnet110': lottery_resnet.wide_resnet110,
        'wide-resnet1202': lottery_resnet.wide_resnet1202
    }
    tinyimagenet_models = {
        'vgg11' : tinyimagenet_vgg.vgg11,
        'vgg11-bn' : tinyimagenet_vgg.vgg11_bn,
        'vgg13' : tinyimagenet_vgg.vgg13,
        'vgg13-bn' : tinyimagenet_vgg.vgg13_bn,
        'vgg16' : tinyimagenet_vgg.vgg16,
        'vgg16-bn' : tinyimagenet_vgg.vgg16_bn,
        'vgg19' : tinyimagenet_vgg.vgg19,
        'vgg19-bn' : tinyimagenet_vgg.vgg19_bn,
        'resnet18' : tinyimagenet_resnet.resnet18,
        'resnet34' : tinyimagenet_resnet.resnet34,
        'resnet50' : tinyimagenet_resnet.resnet50,
        'resnet101' : tinyimagenet_resnet.resnet101,
        'resnet152' : tinyimagenet_resnet.resnet152,
        'wide-resnet18' : tinyimagenet_resnet.wide_resnet18,
        'wide-resnet34' : tinyimagenet_resnet.wide_resnet34,
        'wide-resnet50' : tinyimagenet_resnet.wide_resnet50,
        'wide-resnet101' : tinyimagenet_resnet.wide_resnet101,
        'wide-resnet152' : tinyimagenet_resnet.wide_resnet152,
    }
    imagenet_models = {
        'vgg11' : imagenet_vgg.vgg11,
        'vgg11-bn' : imagenet_vgg.vgg11_bn,
        'vgg13' : imagenet_vgg.vgg13,
        'vgg13-bn' : imagenet_vgg.vgg13_bn,
        'vgg16' : imagenet_vgg.vgg16,
        'vgg16-bn' : imagenet_vgg.vgg16_bn,
        'vgg19' : imagenet_vgg.vgg19,
        'vgg19-bn' : imagenet_vgg.vgg19_bn,
        'resnet18' : imagenet_resnet.resnet18,
        'resnet34' : imagenet_resnet.resnet34,
        'resnet50' : imagenet_resnet.resnet50,
        'resnet101' : imagenet_resnet.resnet101,
        'resnet152' : imagenet_resnet.resnet152,
        'wide-resnet50' : imagenet_resnet.wide_resnet50_2,
        'wide-resnet101' : imagenet_resnet.wide_resnet101_2,
    }
    models = {
        'default' : default_models,
        'lottery' : lottery_models,
        'tinyimagenet' : tinyimagenet_models,
        'imagenet' : imagenet_models
    }
    if model_class == 'imagenet':
        print("WARNING: ImageNet models do not implement `dense_classifier`.")
    return models[model_class][model_architecture]

def pruner(method):
    prune_methods = {
        'rand' :    pruners.Rand,
        'mag' :     pruners.Mag,
        'snip' :    pruners.SNIP,
        'grasp':    pruners.GraSP,
        'synflow' : pruners.SynFlow,
        'mica' : pruners.MinimumConnectionAssurance
    }
    return prune_methods[method]

def optimizer(optimizer):
    optimizers = {
        'adam' : (optim.Adam, {}),
        'sgd' : (optim.SGD, {}),
        'momentum' : (optim.SGD, {'momentum' : 0.9, 'nesterov' : True}),
        'rms' : (optim.RMSprop, {})
    }
    return optimizers[optimizer]

