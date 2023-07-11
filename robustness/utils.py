import numpy as np
import matplotlib.pyplot as plt
import json
import os
from PIL import Image

import torch
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision import utils as tutils


def get_imagenet_data():
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    # https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
    # class_idx = json.load(open("./imagenet_class_index.json"))
    # idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),  # ToTensor : [0, 255] -> [0, 1]
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    la = sorted(os.listdir('/wangrun/lzy/Dataset_raw/imagenet100'))
    imagnet_data = dsets.ImageFolder(root='/wangrun/lzy/Dataset_raw/imagenet100/',
                                     transform=transform,
                                     target_transform=lambda x: la[x])
    data_loader = torch.utils.data.DataLoader(imagnet_data, batch_size=1, shuffle=False)
    print("Used normalization: mean=", MEAN, "std=", STD)
    return iter(data_loader).__next__()


def get_pred(model, images, device):
    logits = model(images.to(device))
    _, pres = logits.max(dim=1)
    return pres.cpu()


def imshow(img, title, pth):
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True)
    npimg = img.numpy()
    fig = plt.figure(figsize=(5, 5))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.savefig(pth)
    # plt.show()


def imsave(img, pth):
    img = torchvision.utils.make_grid(img.cpu().data, normalize=True)
    tutils.save_image(img, pth)


def image_folder_custom_label(root, transform, idx2label):
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']

    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes

    label2idx = {}

    for i, item in enumerate(idx2label):
        label2idx[item] = i
    la = sorted(os.listdir('/wangrun/lzy/Dataset_raw/imagenet100'))
    class_idx = json.load(open("./imagenet_class_index.json"))

    key2idx = {}
    for key, val in enumerate(class_idx):
        key2idx[val[0]] = key

    new_data = dsets.ImageFolder(root=root, transform=transform,
                                 target_transform=lambda x: key2idx[la[x]])
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data


def l2_distance(model, images, adv_images, labels, device="cuda"):
    outputs = model(adv_images)
    _, pre = torch.max(outputs.data, 1)
    corrects = (labels.to(device) == pre)
    delta = (adv_images - images.to(device)).view(len(images), -1)
    l2 = torch.norm(delta[~corrects], p=2, dim=1).mean()
    return l2
