import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
from arguments import get_args
from models import get_model_class
from evaluating import compute_class_accuracy, compute_overall_accuracy
import pickle
from utils import to_cuda, get_dataset, set_seed


def get_optimizer(model, args):

    #get optimizer
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=args.lr,
                              momentum=args.momentum,
                              nesterov=args.nesterov)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               betas=args.beta,
                               eps=args.eps)
    else:
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=args.lr,
                                  alpha=args.alpha,
                                  eps=args.eps)
    return optimizer


def train_student_normal(model, args, trainloader, testloader, seed):

    if torch.cuda.is_available() and args.use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.set_train_mode()

    #get loss function
    criterion = nn.CrossEntropyLoss(reduction='mean')

    optimizer = get_optimizer(model, args)

    loss_values = []
    total_accuracy = []
    epoch_eval = []

    #train the student network
    for epoch in range(args.nr_epochs):
        loss_epoch = 0.0
        for i, data in enumerate(trainloader, 0):
            samples, labels = data
            samples = to_cuda(samples, args.use_cuda)
            labels = to_cuda(labels, args.use_cuda)

            #zero the gradients of network params
            optimizer.zero_grad()

            #define loss
            output_logits = model(samples)
            loss = criterion(output_logits, labels)

            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

        loss_epoch /= float(i)
        loss_values.append(loss_epoch)
        print("Loss at epoch {} is {}".format(epoch, loss_epoch))

        if epoch % args.eval_interval == 0:
            model.eval()
            acc = compute_overall_accuracy(testloader, model, args.use_cuda)
            total_accuracy.append(acc)
            epoch_eval.append(epoch)
            model.train()
            print("Accuracy at epoch {} is {}".format(epoch, acc))

        if epoch % args.save_interval == 0:
            print("Saving model at {} epoch".format(epoch))
            with open(args.dataset +
                      "_student_network_simple" +
                      args.student_model + str(seed) + "_" + str(args.id), "wb") as f:
                torch.save(model.state_dict(), f)

    return epoch_eval, loss_values, total_accuracy


def criterion1(logits_teacher, logits_student, true_labels):

    term1 = F.mse_loss(logits_student, logits_teacher) / (32 * 10)
    term2 = F.cross_entropy(logits_student, true_labels)

    loss_function = 0.9 * term1 + 0.1 * term2

    return loss_function


def criterion2(logits_teacher, logits_student, true_labels):
    
    term1 = F.kl_div(
        F.log_softmax(logits_student / 2, dim=1),
        F.softmax(logits_teacher / 2, dim=1), reduction="batchmean")

    term2 = F.cross_entropy(logits_student, true_labels)

    loss_function = 0.9 * 2 * term1 + 0.1 * term2

    return loss_function


def train_student_teacher(stud_model, teacher_model, args, trainloader, testloader, seed):

    if torch.cuda.is_available() and args.use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    #get the teacher model
    with open(args.dataset + "_teacher_network_" + args.teacher_model + "_" + str(seed), "rb") as f:
        teacher_model.load_state_dict(torch.load(f))

    stud_model.to(device)
    stud_model.set_train_mode()

    teacher_model.to(device)
    teacher_model.set_eval_mode()

    #set optimizer
    optimizer = get_optimizer(stud_model, args)

    loss_values = []
    total_accuracy = []
    epoch_eval = []

    #train the student network
    for epoch in range(args.nr_epochs):
        loss_epoch = 0.0
        for i, data in enumerate(trainloader, 0):
            samples, labels = data
            samples = to_cuda(samples, args.use_cuda)
            labels = to_cuda(labels, args.use_cuda)

            #zero the gradients of network params
            optimizer.zero_grad()

            #define loss
            teacher_output_logits = teacher_model(samples)
            student_output_logits = stud_model(samples)

            loss = criterion2(
                teacher_output_logits, student_output_logits, labels)

            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()

        loss_epoch /= float(i)
        loss_values.append(loss_epoch)
        print("Loss at epoch {} is {}".format(epoch, loss_epoch))

        if epoch % args.eval_interval == 0:
            stud_model.eval()
            acc = compute_overall_accuracy(
                testloader, stud_model, args.use_cuda)
            total_accuracy.append(acc)
            epoch_eval.append(epoch)
            stud_model.train()
            print("Accuracy at epoch {} is {}".format(epoch, acc))

        if epoch % args.save_interval == 0:
            print("Saving model at {} epoch".format(epoch))
            with open(args.dataset +
                      "_student_network_teacher" +
                      args.student_model + str(seed) +'_' + str(args.id), "wb") as f:
                torch.save(stud_model.state_dict(), f)

    return epoch_eval, loss_values, total_accuracy


def train_student():

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

    #get student and teacher models
    student_model_class = get_model_class(args.student_model)
    teacher_model_class = get_model_class(args.teacher_model)
    if "MLP" in args.student_model:
        stud_model_simple = student_model_class(mlp_input_neurons, 10, args.dropout)
        stud_model_teacher = student_model_class(mlp_input_neurons, 10, args.dropout)
        teacher_model = teacher_model_class(mlp_input_neurons, 10, args.dropout)
    else:
        stud_model_simple = student_model_class(nr_channels, 10, args.dropout)
        stud_model_teacher = student_model_class(nr_channels, 10, args.dropout)
        teacher_model = teacher_model_class(nr_channels, 10, args.dropout)

    print("Train student with teacher help")
    loss_epoch2, loss_values2, total_accuracy2 = train_student_teacher(
        stud_model_teacher, teacher_model, args, trainloader, testloader, seed)

    print("Train simple student")
    loss_epoch1, loss_values1, total_accuracy1 = train_student_normal(
        stud_model_simple, args, trainloader, testloader, seed)


    with open("params" + args.dataset + '_' + args.teacher_model + '_' + str(seed), "rb") as f:
        _, epoch_eval_teacher, total_accuracy_teacher = pickle.load(f)


    #plot loss and total accuracy
    plt.figure(1)
    plt.plot(range(0, args.nr_epochs), loss_values1)
    plt.plot(range(0, args.nr_epochs), loss_values2)
    plt.legend(['student_simple', 'student_teacher'], loc='upper right')

    plt.xlabel('Nr Epochs')
    plt.ylabel('Loss function value')
    plt.title('Loss function comparison between students')
    plt.savefig('Loss_function_' + args.dataset +  '_students' + str(seed) + "_" + str(args.id))

    plt.figure(2)
    plt.plot(loss_epoch1, total_accuracy1)
    plt.plot(loss_epoch2, total_accuracy2)
    plt.plot(epoch_eval_teacher, total_accuracy_teacher)
    plt.legend(['student_simple', 'student_teacher', 'teacher'], loc='lower right')

    plt.xlabel('Nr Epochs')
    plt.ylabel('Total accuracy')
    plt.title('Accuracy comparison between students')
    plt.savefig('Accuracy_' + args.dataset + '_students' + str(seed) + "_" + str(args.id))


if __name__ == '__main__':
    train_student()
