import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import os
import argparse
from models import *  # Ensure you have the correct model imports
from utils import progress_bar, accuracy, AverageMeter
from act_final import *  # Ensure you have the correct imports
from experiments.data.dataset.TinyImagenet.tiny_imagnenet import TinyImageNet

parser = argparse.ArgumentParser(description='PyTorch Tiny Imagenet Testing')
parser.add_argument('--save_folder', default='checkpoint', type=str, help='checkpoint save folder')
parser.add_argument('--data_folder', default='/workspace/data', type=str, help='tiny imagenet root folder')
parser.add_argument('--n_classes', default=200, type=int, help='number of classes')
parser.add_argument('--device', default='0', type=str, help='device number')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--model_path', required=True, type=str, help='path to the model checkpoint')

args = parser.parse_args()

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True

def test(net, testloader, criterion):
    net.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            top5.update(prec5.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            losses.update(loss.item(), inputs.size(0))

            bar_text = 'Loss: %.3f | Top5 Acc: %.3f%% | Top1 Acc: %.3f%% (%d/%d)' % (
                losses.avg, top5.avg, top1.avg, top1.avg * top1.count // 100, top1.count)
            progress_bar(batch_idx, len(testloader), bar_text)

    results = {
        'top1': top1.avg,
        'top5': top5.avg,
        'loss': losses.avg,
    }

    return results

def main():
    # Data
    print('==> Preparing data..')
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet mean and std
    ])

    testset =TinyImageNet(root=os.path.join(args.data_folder),train=False,transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    print('==> Building model..')
    net = ResNet18(num_classes=args.n_classes, act=nn.ReLU(), img_size=64)
    net = net.to(device)

    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.model_path), 'Error: no checkpoint file found!'
    checkpoint = torch.load(args.model_path)
    net.load_state_dict(checkpoint['net'])

    criterion = nn.CrossEntropyLoss()

    # Test the model
    print('==> Testing the model..')
    test_results = test(net, testloader, criterion)
    print('Test Results: Top1 Accuracy: %.3f | Top5 Accuracy: %.3f | Loss: %.3f' % (
        test_results['top1'], test_results['top5'], test_results['loss']))

if __name__ == '__main__':
    main()
