from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import util.lr_decay as lrd
import timm

from torchvision import datasets, models, transforms
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
import time
import cv2
import copy
from ggg import EfficientNet
# from coatnet import coatnet_0
from coatnet_re import coatnet_0
from tqdm import tqdm
from timm.data.mixup import Mixup
from tensorboardX import SummaryWriter
from focalloss import FocalLoss
from PolyLoss import PolyLoss
from My_model.CNNSWIN_feafusion import FeaFusion_0
from labelsmooth import LabelSmooth
# from focal_smooth import FocalSmooth
from vgg import vgg16_bn
from models_vit import vit_base_patch16
# from other_model.maxvit import MaxViT
# from other_model.focalnet import FocalNet
from other_model.swin_transformer import SwinTransformer

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.ion()  # interactive mode

writer = SummaryWriter('./log/focalnet/')


def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            pbr = tqdm(range(len(dataloaders[phase])))
            for index, (inputs, labels) in zip(pbr, dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 零参数梯度
                optimizer.zero_grad()

                # 前向
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 后向+仅在训练阶段进行优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                # 显示时间进度轴
                pbr.set_description("loss:{:.3f} correct:{:.2f}%".format(running_loss / (index + 1),
                                                                         100 * running_corrects / (
                                                                             len(dataloaders[phase].dataset))))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 深度复制mo
            if phase == 'val' and epoch_acc > best_acc:
                torch.save(model_conv.state_dict(),
                           '/workspace/cls/mae-main/outputs/sz-10/duibi_cnn+swin/duibi_ouratten/' + str(epoch) + '_w' + '.pth')
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Acc', epoch_acc, epoch)
            else:
                writer.add_scalar('Test/Loss', epoch_loss, epoch)
                writer.add_scalar('Test/Acc', epoch_acc, epoch)

    writer.close()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model


data_transforms = {
    'train': transforms.Compose([
            transforms.Resize(224),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224),
            transforms.RandomRotation(45),
        # RandomResizedCropAndInterpolation(224, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear'),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(224),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# data_dir = '/workspace/cls/WC_ALL/BX/'
# data_dir = '/workspace/cls/WC_ALL/Medicine_img/'
data_dir = '/workspace/cls/WC_ALL/sz/'
# data_dir = '/workspace/cls/Multi_angel/data_view/'

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
print(dataset_sizes)
class_names = image_datasets['train'].classes
torch.manual_seed(1024)
np.random.seed(1024)

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.002)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


feature_extract = True

# model_ft = models.mobilenet_v2(pretrained=False)
# model_ft = coatnet_0()
# # set_parameter_requires_grad(model_ft, feature_extract)
# num_ftrs = model_ft.classifier[1].in_features
# model_ft.classifier[1] = nn.Linear(num_ftrs, 22)

model_name = "FeaFusion_0"
use_pretrained = False
num_classes = 10

if model_name == 'SwinTransformer':
    model_ft = SwinTransformer(
        num_classes=7
    )

if model_name == 'Maxvit':
    model_ft = MaxViT(
        num_classes=7
        )

if model_name == 'focalNet':
    model_ft = FocalNet(
        num_classes=7,
    )

if model_name == "FeaFusion_0":
    model_ft = FeaFusion_0()

if model_name == "vit_mae":
    model_ft = vit_base_patch16(
        num_classes=10,
        drop_path_rate=0.1,
    )

if model_name == "coatnet_0":
    model_ft = coatnet_0()

elif model_name == "vgg":
    model_ft = vgg16_bn()

elif model_name == "effnet":

    model_ft = EfficientNet.from_pretrained('efficientnet-b0', load_weights=True, advprop=False, num_classes=10,
                                            in_channels=3)
elif model_name == "resnet":
    model_ft = models.resnet50()
    # set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)

elif model_name == "densenet":
    model_ft = models.densenet169(pretrained=use_pretrained)
    # set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, num_classes)

elif model_name == "mobilenet_v2":

    model_ft = models.mobilenet_v2(pretrained=use_pretrained)
    # set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes)

# if model_name == "vit_mae":
#     checkpoint = torch.load('/workspace/cls/mae-main/output_dir/pretrained_mae/checkpoint-99.pth', map_location='cpu')
#
#     checkpoint_model = checkpoint['model']
#     state_dict = model_ft.state_dict()
#
#     for k in ['head.weight', 'head.bias']:
#         if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
#             print(f"Removing key {k} from pretrained checkpoint")
#             del checkpoint_model[k]
#     model_ft.load_state_dict(checkpoint_model, strict=False)
#     # print("Load pre-trained checkpoint from: %s" % checkpoint_model)

# if model_name == "FeaFusion_0":
#     if resume:
#         if resume.startswith('https'):
#             checkpoint = torch.hub.load_state_dict_from_url(
#                 args.resume, map_location='cpu', check_hash=True)
#         else:
#             checkpoint = torch.load(resume, map_location='cpu')
#         model_ft.load_state_dict(checkpoint)
#         # start_epoch = checkpoint['epoch'] + 1
#         print("Resume checkpoint %s" % resume)


model_conv = model_ft.to(device)
# criterion = nn.CrossEntropyLoss()
# criterion = LabelSmooth()
criterion = FocalLoss()
# criterion = PolyLoss()
# criterion = FocalSmooth()
# criterion = loss1 + loss2
# optimizer_conv = torch.optim.Adam(model_conv.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
optimizer_conv = torch.optim.AdamW(model_conv.parameters(), lr=0.0001)

# Decay LR by a factor of 0.1 every 7 epochs

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_conv, T_max=200, eta_min=0, last_epoch=-1, verbose=False)
# exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_conv, T_max=100, eta_min=0.001)
model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=200)

output_image = output_image[:, :, ::-1]
cv2.imwrite("output/output_image.jpg", output_image)

torch.save(model_conv.state_dict(), '/workspace/cls/mae-main/outputs/sz-10/duibi_cnn+swin/duibi_ouratten/w.pth')

visualize_model(model_conv)
