import matplotlib.pyplot as plt
import cv2
import os
import sys

import pickle

def make_plot():

    with open("MNIST_MLPRelu", "rb") as f:
        loss_values_relu, epoch_eval_relu, total_accuracy_relu = pickle.load(f)


    with open("MNIST_MLPTan", "rb") as f:
        loss_values_tan, epoch_eval_tan, total_accuracy_tan = pickle.load(f)


    with open("MNIST_MLPSigmoid", "rb") as f:
        loss_values_sigm, epoch_eval_sigm, total_accuracy_sigm = pickle.load(f)


    #plot loss and total accuracy
    plt.figure(1)
    plt.plot(range(0, 1000), loss_values_sigm)
    plt.plot(range(0, 1000), loss_values_relu)
    plt.plot(range(0, 1000), loss_values_tan)
    plt.legend(['sigm', 'relu', 'tan'], loc='upper right')

    plt.xlabel('Nr Epochs')
    plt.ylabel('Loss function value')
    plt.title('Loss function comparison between non-linear activation')
    plt.savefig('Loss_function_MNIST_MLP_SGD_non_linear_activations_comparison')

    plt.figure(2)
    plt.plot(epoch_eval_relu, total_accuracy_relu)
    plt.plot(epoch_eval_sigm, total_accuracy_sigm)
    plt.plot(epoch_eval_tan, total_accuracy_tan)
    plt.legend(['relu', 'sigm', 'tan'], loc='lower right')

    plt.xlabel('Nr Epochs')
    plt.ylabel('Total accuracy')
    plt.title('Accuracy comparison between non-linear activations')
    plt.savefig('Accuracy_MNIST_MLP_SGD_non_linear_activations_comparison')


def make_plot2():

    with open("MNIST_MLPRelu_Adam_0", "rb") as f:
        loss_values_adam, epoch_eval_adam, total_accuracy_adam = pickle.load(f)


    with open("MNIST_MLPRelu_SGD_0", "rb") as f:
        loss_values_sgd, epoch_eval_sgd, total_accuracy_sgd = pickle.load(f)


    with open("MNIST_MLPRelu_RMSprop_0", "rb") as f:
        loss_values_rms, epoch_eval_rms, total_accuracy_rms = pickle.load(f)

    with open("MNIST_MLPTan_SGD_0.9", "rb") as f:
        loss_values_moment, epoch_eval_moment, total_accuracy_moment = pickle.load(f)


    #plot loss and total accuracy
    plt.figure(1)
    plt.plot(range(0, 300), loss_values_sgd)
    plt.plot(range(0, 300), loss_values_adam)
    plt.plot(range(0, 300), loss_values_moment)
    plt.plot(range(0, 300), loss_values_rms)
    plt.legend(['sgd', 'adam', 'momentum-sgd', "rmsprop"], loc='upper right')

    plt.xlabel('Nr Epochs')
    plt.ylabel('Loss function value')
    plt.title('Loss function comparison between optimization algorithms using MLP')
    plt.savefig('Loss_function_MNIST_MLP_optimization_algorithms_relu')

    plt.figure(2)
    plt.plot(epoch_eval_adam, total_accuracy_adam)
    plt.plot(epoch_eval_sgd, total_accuracy_sgd)
    plt.plot(epoch_eval_moment, total_accuracy_moment)
    plt.plot(epoch_eval_rms, total_accuracy_rms)

    plt.legend(['adam', 'sgd', 'momentum-sgd', "rmsprop"], loc='lower right')

    plt.xlabel('Nr Epochs')
    plt.ylabel('Total accuracy')
    plt.title('Accuracy comparison between optimization algorithms using MLP')
    plt.savefig('Accuracy_MNIST_MLP_optimization_algorithms_relu')



def make_plot3():

    with open("Cifar10_CNNSimple_SGD_0", "rb") as f:
        lv_cnn_simple, ev_cnn_simple, acc_cnn_simple = pickle.load(f)


    with open("Cifar10_CNNDropout_SGD_0", "rb") as f:
        lv_cnn_dropout, ev_cnn_dropout, acc_cnn_dropout = pickle.load(f)


    with open("Cifar10_CNNBatchNorm_SGD_0", "rb") as f:
        lv_cnn_bnorm, ev_cnn_bnorm, acc_cnn_bnorm = pickle.load(f)

    with open("Cifar10_CNNSimple_Adam_0", "rb") as f:
        lv_cnn_simple_adam, ev_cnn_simple_adam, acc_cnn_simple_adam = pickle.load(f)


    #plot loss and total accuracy
    plt.figure(1)
    plt.plot(range(0, 300), lv_cnn_simple)
    plt.plot(range(0, 300), lv_cnn_dropout)
    plt.plot(range(0, 300), lv_cnn_bnorm)
    #plt.plot(range(0, 300), lv_cnn_simple_adam)
    plt.legend(['cnn_simple', 'dropout', 'batch_norm'], loc='upper right')

    plt.xlabel('Nr Epochs')
    plt.ylabel('Loss function value')
    plt.title('Loss function comparison between regularization algo CNN')
    plt.savefig('Loss_function_CIFAR10_CNN_reg_alg')

    plt.figure(2)
    plt.plot(ev_cnn_simple, acc_cnn_simple)
    plt.plot(ev_cnn_dropout, acc_cnn_dropout)
    plt.plot(ev_cnn_bnorm, acc_cnn_bnorm)
    #plt.plot(ev_cnn_simple_adam, acc_cnn_simple_adam)

    plt.legend(['cnn_simple', 'dropout', 'batch_norm'], loc='lower right')

    plt.xlabel('Nr Epochs')
    plt.ylabel('Total accuracy')
    plt.title('Accuracy comparison between regularization algo CNN')
    plt.savefig('Accuracy_CIFAR10_CNN_reg_algo_sgd')





if __name__ == '__main__':
    make_plot3()
