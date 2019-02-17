import argparse

def get_args():
    parser = argparse.ArgumentParser(description='DM')
    parser.add_argument(
        '--teacher_model',
        default='MLPTan',
        help='nn model to use:MLPTan, MLPRelu, MLPSigmoid, CNNSimple, CNNDropout, CNNBatchNorm, CNNDense, CNNFUll')
    parser.add_argument(
        '--student_model',
        default='MLPTan',
        help='nn model to use:MLPTan, MLPRelu, MLPSigmoid, CNNSimple, CNNDropout, CNNBatchNorm, CNNDense, CNNFUll')
    parser.add_argument(
        "--optimizer",
        default="SGD",
        help='optimizer to use: SGD, Adam, RMSprop',)
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='learning rate (default: 7e-4)')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0,
        help='momentum for SGD')
    parser.add_argument(
        '--nesterov',
        action='store_true',
        default=False,
        help="Neserov momentum for SGD")
    parser.add_argument(
        '--eps',
        type=float,
        default=1e-5,
        help='optimizer epsilon (default: 1e-5)')
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--beta',
        type=tuple,
        default=(0.9, 0.99),
        help='Adam optimizer beta (default: (0.9, 0.99)')
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Dropout value (default 0.5)')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='random seed (default: 1)')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--use_cuda',
        action='store_true',
        default=True,
        help='use cuda for training'
    )
    parser.add_argument(
        "--nr_epochs",
        type=int,
        default=1000,
        help='number of epochs to train')
    parser.add_argument(
        '--num-processes',
        type=int,
        default=8,
        help='how many training CPU processes to use (default: 16)')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='batch-size (default: 32)')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save_interval',
        type=int,
        default=10,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval_interval',
        type=int,
        default=5,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--dataset',
        default='Cifar10',
        help='datset to be used: MNIST, Cifar10, Cifar100')

    parser.add_argument(
        '--id',
        default=10,
        help='id for experiment'
    )

    args = parser.parse_args()
    return args

