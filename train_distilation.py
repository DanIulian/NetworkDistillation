import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from arguments import get_args
from models import get_model_class
from evaluating import compute_class_accuracy, compute_overall_accuracy
import pickle

from utils import to_cuda, get_dataset, set_seed


def train_teacher():

    #get args
    args = get_args()
    seed = set_seed(args.seed, args.use_cuda)

    trainset, testset, nr_channels, mlp_input_neurons, classes = get_dataset(args)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_processes)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size,
        shuffle=False, num_workers=1)

    #get teacher model
    teacher_model_class = get_model_class(args.teacher_model)
    if "MLP" in args.teacher_model:
        teacher_model = teacher_model_class(mlp_input_neurons, 10, args.dropout)
    else:
        teacher_model = teacher_model_class(nr_channels, 10, 6)


    if torch.cuda.is_available() and args.use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    teacher_model.to(device)
    teacher_model.train()

    #get loss function
    criterion = nn.CrossEntropyLoss(reduction='mean')

    #get optimizer
    if args.optimizer == "SGD":
        optimizer = optim.SGD(teacher_model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              nesterov=args.nesterov,
                              weight_decay=0.0001)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(teacher_model.parameters(),
                               lr=args.lr,
                               betas=args.beta,
                               eps=args.eps,
                               weight_decay=0.0001)
    else:
        optimizer = optim.RMSprop(teacher_model.parameters(),
                                  lr=args.lr,
                                  alpha=args.alpha,
                                  eps=args.eps)

    loss_values = []
    total_accuracy = []
    epoch_eval = []



    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[30, 60, 90, 120, 150, 180], gamma=0.1)

    #train the teacher network
    for epoch in range(args.nr_epochs):
        loss_epoch = 0.0
        scheduler.step()
        for i, data in enumerate(trainloader, 0):
            samples, labels = data
            samples = to_cuda(samples, args.use_cuda)
            labels = to_cuda(labels, args.use_cuda)

            #zero the gradients of network params
            optimizer.zero_grad()

            #define loss
            output_logits = teacher_model(samples)
            loss = criterion(output_logits, labels)

            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

        loss_epoch /= float(i)
        loss_values.append(loss_epoch)
        print("Loss at epoch {} is {}".format(epoch, loss_epoch))

        if epoch % args.eval_interval == 0:
            teacher_model.eval()
            acc = compute_overall_accuracy(testloader, teacher_model, args.use_cuda)
            total_accuracy.append(acc)
            epoch_eval.append(epoch)
            teacher_model.train()
            print("Accuracy at epoch {} is {}".format(epoch, acc))

        if epoch % args.save_interval == 0:
            print("Saving model at {} epoch".format(epoch))
            with open(args.dataset + "_teacher_network_" + args.teacher_model + "_" + str(seed), "wb") as f:
                torch.save(teacher_model.state_dict(), f)

    #plot loss and total accuracy
    plt.figure(1)
    plt.plot(loss_values)
    plt.xlabel('Nr Epochs')
    plt.ylabel('Loss function')
    plt.title('Loss function for Teacher on' + args.dataset + " using " + args.teacher_model)
    plt.savefig('Loss_function_teacher' + args.teacher_model + "_" + args.dataset + str(seed))

    plt.figure(2)
    plt.plot(epoch_eval, total_accuracy)
    plt.xlabel('Nr Epochs')
    plt.ylabel('Total accuracy')
    plt.title('Accuracy for Teacher on ' + args.dataset + " using " + args.teacher_model)
    plt.savefig('Accuracy_teacher' + args.teacher_model + "_" + args.dataset + str(seed))

    with open("params" + args.dataset + '_' + args.teacher_model + '_' + str(seed), "wb") as f:
        params = [
            loss_values, epoch_eval, total_accuracy
        ]
        pickle.dump(params, f)


if __name__ == '__main__':
    train_teacher()
