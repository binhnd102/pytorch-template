from __future__ import print_function
import time
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from warmup_scheduler import GradualWarmupScheduler

from utils import TwoCropTransform, AverageMeter, save_checkpoint, get_learning_rate, accuracy
from dataset.image_folder import DatasetManager
from model.supcon_resnet import SupConResNet
from model.efficientnet import SupConEfficient
from losses.sup_contrastive import SupConLoss


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    current_LR = get_learning_rate(optimizer)[0]
    # switch to train mode
    model.train()
    for batch, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = torch.cat([images[0], images[1]], dim=0)
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        # compute losses
        features = model(images, contrastive=True)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, labels)

        losses.update(loss.item(), features.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch % print_freq == 0 and verbose == True:
            print('Epoch: [{0}/{1}][{2}/{3}]\t'
                  'LR: {LR:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                epoch, epochs, batch, len(train_loader), LR=current_LR, batch_time=batch_time,
                data_time=data_time, loss=losses))
    return losses.avg


# hyper-parameters
bs = 4
num_workers = 2
epochs = 50
size = 224
temperature = 0.07
data_dir = 'flower_photos'
lr = 0.05
momentum = 0.9
weight_decay = 1e-5
epochs = 50
print_freq = 5
verbose = 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
best_err1 = 100
best_err5 = 100
arch_name = 'efficientnet-b0'

# Transform for constrastive learning
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=size, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomErasing()
])
train_transform = TwoCropTransform(train_transform)
train_dataset, valid_dataset = DatasetManager(data_dir, train_transform=train_transform).split()
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True, drop_last=True)

model = SupConEfficient(arch_name, num_classes=5)
model = model.to(device)
# define loss function (criterion) and optimizer
criterion = SupConLoss(temperature).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr,
                            momentum=momentum,
                            weight_decay=weight_decay)

scheduler_steplr = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)


train_loss = None
best_loss = 100
for epoch in range(0, epochs):
    scheduler_warmup.step(epoch, train_loss)
    # train for one epoch
    train_loss = train(train_loader, model, criterion, optimizer, epoch)

    # remember best prec@1 and save checkpoint
    is_best = train_loss <= best_loss
    best_loss = min(train_loss, best_loss)

    # early stopping
    patient = 3
    if is_best:
        best_loss = train_loss
        patient = 3
    else:
        patient -= 1

    if patient == 0:
        break

    print('Current best train loss:', best_loss)
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, is_best)


print('Current best train loss:', best_loss)