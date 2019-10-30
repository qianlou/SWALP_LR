import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data_utils

num_classes_dict = {
    "CIFAR10":10,
    "CIFAR100":100,
    "DOGFISH":1,
}

def get_data(dataset, data_path, batch_size, num_workers):
    assert dataset in ["CIFAR10", "CIFAR100","DOGFISH"]
    print('Loading dataset {} from {}'.format(dataset, data_path))
    if dataset in ["CIFAR10",  "CIFAR100"]:
        ds = getattr(datasets, dataset.upper())
        path = os.path.join(data_path, dataset.lower())
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = ds(path, train=True, download=True, transform=transform_train)
        #print("here is get_data")
        #print(train_set)
        val_set = ds(path, train=True, download=True, transform=transform_test)
        test_set = ds(path, train=False, download=True, transform=transform_test)
        train_sampler = None
        val_sampler = None
    elif dataset in ["DOGFISH"]:
        path = os.path.join(data_path, dataset.lower())
        
        import numpy as np
        FilePath="./features/dog_fish_features.npz"
        print('Loading dogfish features from disk...')
        f=np.load(FilePath)
        X_train = f['X_train']
        X_test = f['X_test']
        Y_train = f['Y_train']
        print(Y_train)
        Y_test = f['Y_test']
        #print(X_train.shape)
        X_train = torch.Tensor(list(X_train))
        Y_train = torch.Tensor(list(Y_train))
        X_test = torch.Tensor(list(X_test))
        Y_test = torch.Tensor(list(Y_test))
        train_set = data_utils.TensorDataset(X_train, Y_train)
        test_set  = data_utils.TensorDataset(X_test, Y_test)
        val_set   = data_utils.TensorDataset(X_train, Y_train)
        train_sampler = None
        val_sampler = None
    else:
        raise Exception("Invalid dataset %s"%dataset)

    loaders = {
        'train': torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
       }

    return loaders

'''
def get_data(dataset, data_path, batch_size, num_workers):
    assert dataset in ["CIFAR10", "CIFAR100"]
    print('Loading dataset {} from {}'.format(dataset, data_path))
    if dataset in ["CIFAR10",  "CIFAR100"]:
        ds = getattr(datasets, dataset.upper())
        path = os.path.join(data_path, dataset.lower())
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_set = ds(path, train=True, download=True, transform=transform_train)
        print("here is get_data")
        print(train_set)
        val_set = ds(path, train=True, download=True, transform=transform_test)
        test_set = ds(path, train=False, download=True, transform=transform_test)
        train_sampler = None
        val_sampler = None
    else:
        raise Exception("Invalid dataset %s"%dataset)
    

    loaders = {
        'train': torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True
        ),
        'val': torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
       }

    return loaders
'''
