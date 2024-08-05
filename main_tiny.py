'''Train Tiny-Imagenet with PyTorch.'''
import sys
sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import os
import argparse

from models import *
from utils import progress_bar, accuracy, AverageMeter
from act_final import *
from experiments.data.dataset.TinyImagenet.tiny_imagnenet import TinyImageNet


# extenal git
# from experiments.activation.acts_swish import *

parser = argparse.ArgumentParser(description='PyTorch Tiny Imagenet Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--save_folder', default='checkpoint', type=str, help='checkpoint save folder')
parser.add_argument('--data_folder', default='/workspace/data', type=str, help='tiny imagenet root folder')
parser.add_argument('--n_classes', default=200, type=int, help='number of classes')
parser.add_argument('--device', default='0', type=str, help='device number')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--repeat', default=1, type=int, help='number of repetitive training')
parser.add_argument('--epoch', default=200, type=int, help='max epoch')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--size', default=1., type=float, help='shufflenet networks size')


args = parser.parse_args()

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    cudnn.benchmark = True

# Training
def train(epoch,net,act_name,trainloader,optimizer,criterion,repeat):
    print(f'\nEpoch: {epoch} - {act_name}')
    net.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


        prec1,prec5 = accuracy(outputs,targets,topk=(1,5))
        top5.update(prec5.item(),inputs.size(0))
        top1.update(prec1.item(),inputs.size(0))
        losses.update(loss.item(),inputs.size(0))
        

        bar_text = '(%d)Loss: %.3f | Top5 Acc: %.3f%% Top1 Acc: %.3f%% (%d/%d)' % (repeat, losses.avg, top5.avg, top1.avg, top1.avg*top1.count//100, top1.count)
        progress_bar(batch_idx, len(trainloader), bar_text)
    
    results = {
        'loss':losses.avg,
        'top1':top1.avg,
        'top5':top5.avg 
    }

    return results

def test(net,testloader,criterion,repeat):
    net.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            prec1,prec5 = accuracy(outputs,targets,topk=(1,5))
            top5.update(prec5.item(),inputs.size(0))
            top1.update(prec1.item(),inputs.size(0))
            losses.update(loss.item(),inputs.size(0))

            bar_text = '(%d)Loss: %.3f | Top5 Acc: %.3f%% Top1 Acc: %.3f%% (%d/%d)' % (repeat, losses.avg, top5.avg, top1.avg, top1.avg*top1.count//100, top1.count)
            progress_bar(batch_idx, len(testloader), bar_text)

    results = {
        'top1':top1.avg,
        'top5':top5.avg, 
        'loss':losses.avg,
    }

    return results 

def main(act,act_name,i):
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    dataset_class = TinyImageNet
    save_folder = args.save_folder
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
        print('make dirs ',save_folder)
    log_dir_name = args.save_folder.split('/')[1]
    log_dir=os.path.join('logs', f'{log_dir_name}',f'{act_name}_{i}')
    print('log_dir : ',log_dir)
    writer = SummaryWriter(logdir=log_dir)

    optimizer = None
    # Data
    print(device)
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet mean and std
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # ImageNet mean and std
    ])

    trainset =dataset_class(root=os.path.join(args.data_folder),train=True,transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testset =dataset_class(root=os.path.join(args.data_folder),train=False,transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=4)


    # Model
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # net.fc = nn.Linear(net.fc.in_features, args.n_classes) 
    net = ResNet18(num_classes=args.n_classes,act=act,img_size=64)
    # net = SENet18()
    # net = shufflenet_tiny(net_size=args.size,n_classes=args.n_classes,act=act)
    # net = EfficientNetB0()

    # net = nn.DataParallel(net,[0,1],0)
    net = net.to(device)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    model_type = net.__class__.__name__
    print(f'{model_type} train') 

    best_acc,best_state = -1.,{}
    best_model_file_name = ""

    for epoch in range(start_epoch, start_epoch+args.epoch):
        train_re = train(epoch,net,act_name,trainloader,optimizer,criterion,i)
        val_re = test(net,testloader,criterion,i)
        scheduler.step()
        if val_re['top1'] > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'epoch': epoch,
                'act' : act,
                'top1':val_re['top1'],
                'top5':val_re['top5']
            }
            if best_model_file_name:
                try:
                    # Attempt to remove the previous best model's checkpoint
                    os.remove(f'./{save_folder}/{best_model_file_name}')
                except:
                    pass
            best_model_file_name = f'{i}_ckpt_{model_type}_{dataset_class.__name__}_{act_name}_{val_re["top1"]:.2f}.pth'
            torch.save(state, f'./{save_folder}/{best_model_file_name}')
            best_acc = val_re['top1']
            best_state = state
        # Logging to TensorBoard
        writer.add_scalar('Loss/train', train_re['loss'], epoch)
        writer.add_scalar('Accuracy-top1/train', train_re['top1'], epoch)
        writer.add_scalar('Loss/val', val_re['loss'], epoch)
        writer.add_scalar('Accuracy-top1/val', val_re['top1'], epoch)
        if args.n_classes == 100:
            writer.add_scalar('Accuracy-top5/val', val_re['top5'], epoch)
    print(best_state['act'],' ',best_state['epoch'],' ',best_state['top1'])

if __name__ == '__main__':
    acts = get_activations(return_type='dict') # name,class()
    acts = {
        'ReLU':nn.ReLU(),
        # 'Swish':Swish(),
        # 'SwishT_C':SwishT_C(),
    }
    for i in range(1,args.repeat+1):
        for name, activation_fn in acts.items():
            print(f'act : {name}')
            main(activation_fn, name,i)
    torch.cuda.empty_cache()

