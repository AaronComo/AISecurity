import timm

import torch
import torch.nn as nn
import torchattacks

import random
import json
from PIL import Image
from torchvision import models
from robustbench.utils import clean_accuracy

from torchattacks import PGD
from .utils import imshow, get_pred, imsave

import torchvision.transforms as transforms

# import argparse 

# parser = argparse.ArgumentParser()

# parser.add_argument('', type=int, help='Radius of cylinder')
# parser.add_argument('name', type=str, choices=['resnet50', 'cspresnet50', 'efficientnet_b0', 'xception',
#                                                'densenet121', 'fbnetc_100', 'mobilenetv2_100', 'resnext101_32x8d'], help='robust model')
# args = parser.parse_args()

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),  # ToTensor : [0, 255] -> [0, 1]
    transforms.Normalize(mean=MEAN, std=STD)
])


def attack(model, images, target):
    # print(f'images:{images} target:{target}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    atk = PGD(model, eps=8 / 255, alpha=2 / 225, steps=10, random_start=True)
    atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    atk.set_mode_targeted_by_label(quiet=True)
    target_labels = target
    adv_images = atk(images, target_labels)

    idx = 0
    pre = get_pred(model, adv_images[idx:idx + 1], device)
    pth = f'./static/robustness/target_{target}.png'
    pth1 = './static/robustness/adversarial.png'
    imshow(adv_images[idx:idx + 1], title="True:{}, Pre:{}".format(idx2label[labels[idx]], idx2label[pre]), pth=pth)
    imsave(adv_images, pth=pth1)
    return pth, pth1, idx2chinese[labels[idx]], idx2chinese[pre]


def test_robust(name, adv_images):
    adv_images = transform(adv_images).unsqueeze(0)  # toTensor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(name, pretrained=True).to(device).eval()
    idx = 0

    pre = get_pred(model, adv_images[idx:idx + 1], device)
    pth = f'./static/robustness/robust_{name}.png'
    imshow(adv_images[idx:idx + 1], title="True:{}, Pre:{}".format(idx2label[labels[idx]], idx2label[pre]), pth=pth)
    return pth, idx2chinese[pre]


def adv_attack(name, x, y_true):
    r"""
    args: attack_model, image, real_label
    name:
    resnet50
    cspresnet50
    efficientnet_b0
    xception
    densenet121
    fbnetc_100
    mobilenetv2_100
    resnext101_32x8d
    """
    global idx2label, labels, idx2chinese
    # images, labels = get_imagenet_data()

    img = Image.open(x)
    img = transform(img)
    img = img.unsqueeze(0)
    images = img
    labels = [y_true]

    class_idx = json.load(open("./robustness/imagenet_class_index.json"))
    chinese_class = open("./robustness/idx2chinese.txt", "r", encoding="utf-8").readlines()
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    idx2chinese = [chinese_class[i].split('\n')[0] for i in range(len(chinese_class))]

    key2idx = {}
    for key, val in class_idx.items():
        key2idx[val[0]] = key
    labels = [int(key2idx[k]) for k in labels]
    labels = torch.tensor(labels)
    # device = "cuda"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True).to(device).eval()
    # model = Robustmodel(id='Standard_R50', dataset='imagenet', threat_model='corruptions').to(device).eval()
    # acc = clean_accuracy(model, images.to(device), labels.to(device))
    print('[Model loaded]')
    # print('Acc: %2.2f %%'%(acc*100))

    # attacklabel, you can choose your own attack label
    target_label = random.randint(0, 999)
    target_label = torch.tensor([target_label])

    return attack(model, images, target_label)
