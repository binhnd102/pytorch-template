import time
import numpy as np
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from warmup_scheduler import GradualWarmupScheduler
from dataset.image_folder import DatasetManager
from utils import AverageMeter, get_learning_rate, accuracy, save_checkpoint
from model.efficientnet import SupConEfficient

lr = 3e-4
momentum = 0.9
weight_decay = 1e-5
epochs = 90
beta = 1.0
cutmix_prob = 0.5
print_freq = 5
verbose = 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
best_err1 = 100
arch_name = 'efficientnet-b0'
checkpoint_path = ''

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# define dataloader
train_dataset, valid_dataset = DatasetManager('./data/flower_photos').split()
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, drop_last=True)
valid_loader = DataLoader(train_dataset, batch_size=12 * 4, shuffle=True)
# create model
model = SupConEfficient(arch_name, num_classes=5)
# model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
# model.freeze_encoder()
model = model.to(device)
print(model)
print('the number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
scheduler_steplr = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)




def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    # switch to train mode
    model.train()
    for batch, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.to(device)
        target = target.to(device)

        r = np.random.rand(1)
        if beta > 0 and r < cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(input.size()[0]).to(device)
            target_a = target
            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            # compute output
            output = model(input)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)

        # measure accuracy and record loss
        err1 = accuracy(output.data, target, topk=(1,))

        losses.update(loss.item(), input.size(0))
        top1.update(err1.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if epoch % print_freq == 0 and verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})'.format(
                epoch, epochs, batch, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion, epoch=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        err1 = accuracy(output.data, target, topk=(1,))

        losses.update(loss.item(), input.size(0))

        top1.update(err1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and verbose == True and epoch != None:
            print('Test (on val set): [{0}/{1}][{2}/{3}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1-err {top1.val:.4f} ({top1.avg:.4f})'.format(
                epoch, epochs, i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1))

    print('* Epoch: [{0}/{1}]\t Top 1-err {top1.avg:.3f}\t Test Loss {loss.avg:.3f}'.format(
        epoch, epochs, top1=top1, loss=losses))
    return top1.avg, losses.avg


# _, _, val_loss = validate(valid_loader, model, criterion)
val_loss = 5
for epoch in range(0, epochs):

    scheduler_warmup.step(epoch, val_loss)
    # train for one epoch
    train_loss = train(train_loader, model, criterion, optimizer, epoch)

    # evaluate on validation set
    err1, val_loss = validate(valid_loader, model, criterion, epoch)

    # remember best prec@1 and save checkpoint
    is_best = err1 <= best_err1
    best_err1 = min(err1, best_err1)

    print('Current best accuracy (top-1):', best_err1)
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'best_err1': best_err1,
        'optimizer': optimizer.state_dict(),
    }, is_best)


print('Best accuracy (top-1 and 5 error):', best_err1)