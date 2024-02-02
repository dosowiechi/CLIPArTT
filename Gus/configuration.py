import argparse

def argparser():
    parser = argparse.ArgumentParser()

    # Directories
    parser.add_argument('--root', type=str, default='/home/vhakim/projects/def-ibenayed/vhakim/ClusT3',
                        help='Base path')
    parser.add_argument('--dataroot', type=str, default='/home/vhakim/GAVH/ETS/Datasets/VisdaC/')
    parser.add_argument('--save', type=str, default='work/', help='Path for base training weights')
    parser.add_argument('--livia', action='store_true', help='To use LIVIA servers directories')

    # General settings
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    # Dataset
    parser.add_argument('--dataset', type=str, default='visda', choices=('cifar10', 'cifar100', 'visda', 'office'))
    parser.add_argument('--split', type=str, default='val', choices=('train', 'val'))
    parser.add_argument('--workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--category', type=str, default='Real World', help='Domain category (OfficeHome)',
                        choices=('Art', 'Clipart', 'Product', 'Real World'))

    #Training/adaptation
    parser.add_argument('--adapt', action='store_true', help='To adapt or not')
    parser.add_argument('--model', type=str, default='RN50', help='Adaptation method',choices=('RN50', 'RN101', 'ViT-B/16', 'ViT-B/32'))
    parser.add_argument('--method', type=str, default='clipart', help='Adaptation method', choices=('tent', 'lame', 'ptbn', 'clipart'))
    parser.add_argument('--batch-size', type=int, default=50, help='Number of images per batch')
    parser.add_argument('--niter', type=int, default=50, help='Number of iterations of adaptation')
    parser.add_argument('--th', type=float, default=1.0, help='Confidence threshold')
    parser.add_argument('--lr', type=float, default=0.001, help='Confidence threshold')
    parser.add_argument('--K', type=int, default=3, help='Number of classes to choose from when adapting')
    parser.add_argument('--mode', type=str, default='norm', help='Parameters to adapt', choices=('norm', 'all', 'batch-adapter', 'conv-adapter'))
    parser.add_argument('--kernel', type=str, default='knn', help='Affinity kernels for LAME', choices=('knn', 'rbf', 'linear'))

    # Distributed
    parser.add_argument('--distributed', action='store_true', help='Activate distributed training')
    parser.add_argument('--init-method', type=str, default='tcp://127.0.0.1:3456', help='url for distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str, help='distributed backend')
    parser.add_argument('--world-size', type=int, default=1, help='Number of nodes for training')

    return parser.parse_args()