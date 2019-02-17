import torch
import torchvision
import torchvision.transforms as transforms


def transform(nr_channels=3, train=False):

    if train:
        trans = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(size=32, padding=4),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5]*nr_channels, std=[0.5]*nr_channels)])
    else:
        trans = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*nr_channels, std=[0.5]*nr_channels)])
    return trans


def to_cuda(data, use_cuda):
    if use_cuda:
        input_ = data.cuda()
    return input_


def set_seed(seed, use_cuda=True):
    import random
    import numpy as np

    if seed == 0:
        seed = random.randint(1, 9999999)

    print("Seed is {}".format(seed))
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    return seed


def get_dataset(args):

    if args.dataset == "MNIST":
        nr_channels = 1
        mlp_input_neurons = 784
        transf = transform(nr_channels)
        trainset = torchvision.datasets.MNIST(
            root='./mnist_data',
            train=True,
            download=True,
            transform=transf
        )
        testset = torchvision.datasets.MNIST(
            root='./mnist_data',
            train=False,
            download=True,
            transform=transf
        )
        classes = ('zero', 'one', 'two', 'three',
                   'four', 'five', 'six', 'seven', 'eight', 'nine')

    elif args.dataset == "Cifar10":
        nr_channels = 3
        mlp_input_neurons = 3072
        transf_train = transform(nr_channels, True)
        transf_test =transform(nr_channels)
        trainset = torchvision.datasets.CIFAR10(
            root='./cifar10_data',
            train=True,
            download=True,
            transform=transf_train
        )
        testset = torchvision.datasets.CIFAR10(
            root='./cifar10_data',
            train=False,
            download=True,
            transform=transf_test
        )
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    else:
        print("No such datset")
        exit(-1)

    return trainset, testset, nr_channels, mlp_input_neurons, classes
