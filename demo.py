import torch
import torch.nn.functional as F
import torchvision

from arguments import get_args
from models import get_model_class
import matplotlib.pyplot as plt

from utils import to_cuda, get_dataset, set_seed
import numpy as np


def to_cuda(data, use_cuda):
    if use_cuda:
        input_ = data.cuda()
    return input_


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg)
    plt.show()


def show_results(testloader, model, classes, use_cuda=True):

    counter = 10
    with torch.no_grad():
        for data in testloader:

            images, labels = data
            labels = labels.numpy()
            print("True Labels:")
            print(' '.join('%5s' % classes[labels[j]] for j in range(5)))

            images_inference = to_cuda(images, use_cuda)

            outputs = F.softmax(model(images_inference), dim=1)

            _, predicted = torch.max(outputs, dim=1)
            predicted = predicted.cpu().numpy()

            print("Predicted Labels:")
            print(' '.join('%5s' % classes[predicted[j]] for j in range(5)))

            imshow(torchvision.utils.make_grid(images))

            counter -= 1
            if counter == 0:
                break


def eval_networks():
    # get args
    args = get_args()
    seed = set_seed(args.seed, args.use_cuda)

    _, testset, nr_channels, mlp_input_neurons, classes = get_dataset(args)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=5,
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
    show_results(testloader, teacher_model, classes,  use_cuda=True)

    print("Eval student model simple")
    #show_results(testloader, stud_model_simple, classes, use_cuda=True)

    print("Eval student model twacher")
    #show_results(testloader, stud_model_teacher, classes,  use_cuda=True)



if __name__ == '__main__':
    eval_networks()
