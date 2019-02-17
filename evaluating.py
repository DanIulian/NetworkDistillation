import torch
import torch.nn.functional as F


import matplotlib.pyplot as plt

from arguments import get_args
from models import get_model_class

from utils import to_cuda, get_dataset, set_seed

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import pylab


def to_cuda(data, use_cuda):
    if use_cuda:
        input_ = data.cuda()
    return input_


def compute_overall_accuracy(testloader, model, use_cuda=True):

    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = to_cuda(images, use_cuda)
            labels = to_cuda(labels, use_cuda)

            outputs = F.softmax(model(images), dim=1)

            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(total, correct)
    return float(correct) / total


def compute_class_accuracy(testloader, model, name, use_cuda=True):

    class_correct, class_total = [0] * 10, [0] * 10
    confusion_matrix = np.zeros((10, 10), dtype=np.int32)

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = to_cuda(images, use_cuda)
            labels = to_cuda(labels, use_cuda)

            outputs = F.softmax(model(images), dim=1)

            _, predicted = torch.max(outputs, dim=1)

            for i in range(predicted.shape[0]):
                class_total[labels[i].item()] += 1
                if labels[i].item() == predicted[i].item():
                    class_correct[predicted[i].item()] += 1
                confusion_matrix[predicted[i].item(), labels[i].item()] += 1

    save_confusion_matrix(confusion_matrix, name)

    print(sum(class_correct))
    return class_correct, confusion_matrix



def save_confusion_matrix(conf_matrix, name):

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(conf_matrix), cmap=plt.cm.jet,
                    interpolation='nearest')

    width, height = conf_matrix.shape

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_matrix[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    plt.savefig(name + '.png', format='png')


def eval_networks():
    # get args
    args = get_args()
    seed = set_seed(args.seed, args.use_cuda)

    _, testset, nr_channels, mlp_input_neurons, classes = get_dataset(args)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size,
        shuffle=False, num_workers=1)

    # get student and teacher models
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

    if torch.cuda.is_available() and args.use_cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    with open(args.dataset + "_teacher_network_" + args.teacher_model + "_" + str(seed), "rb") as f:
        teacher_model.load_state_dict(torch.load(f))

    with open(args.dataset + "_student_network_simple" + args.student_model +str(seed) + "_10", "rb") as f:
        stud_model_simple.load_state_dict(torch.load(f))

    with open(args.dataset + "_student_network_teacher" + args.student_model + str(seed) + "_10", "rb") as f:
        stud_model_teacher.load_state_dict(torch.load(f))

    stud_model_simple.to(device)
    stud_model_teacher.to(device)
    teacher_model.to(device)

    stud_model_simple.eval()
    stud_model_teacher.eval()
    teacher_model.eval()

    print("Eval teacher model")
    compute_class_accuracy(testloader, teacher_model, "ConfusionMatrixTeacherCIFAR110", use_cuda=True)

    print("Eval student model simple")
    compute_class_accuracy(testloader, stud_model_simple, "ConfusionMatrixStudentSCIFAR110", use_cuda=True)

    print("Eval student model twacher")
    compute_class_accuracy(testloader, stud_model_teacher, "ConfusionMatrixStudentTCIFAR110", use_cuda=True)



if __name__ == '__main__':
    eval_networks()
