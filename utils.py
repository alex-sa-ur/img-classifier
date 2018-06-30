import json
import numpy as np
import torch

from PIL import Image
from torchvision import datasets, transforms

pre_re_size = 256

def create_dataset(image_path, re_size, norm_mean, norm_stdv, batch):
    if 'train' in image_path:
        transf = transforms.Compose([
            transforms.RandomResizedCrop(re_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = norm_mean,
                std  = norm_stdv
            )
        ])
    else:
        transf = transforms.Compose([
            transforms.Resize(re_size),
            transforms.CenterCrop(re_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = norm_mean,
                std  = norm_stdv
            )
        ])
        
    img_dataset = datasets.ImageFolder(image_path, transf)
    
    return img_dataset

def create_loader(image_path, re_size, norm_mean, norm_stdv, batch):
    img_dataset = create_dataset(image_path, re_size, norm_mean, norm_stdv, batch)
    img_loader  = torch.utils.data.DataLoader(img_dataset, batch_size = batch, shuffle = True)
    
    return img_loader

def process_image(image_path, re_size, norm_means, norm_stdv):
    img = Image.open(image_path)
    img = img.resize((pre_re_size,int(pre_re_size*img.size[1]/img.size[0])))
    img = img.crop((
        img.size[0]/2 - re_size/2,
        img.size[1]/2 - re_size/2,
        img.size[0]/2 + re_size/2,
        img.size[1]/2 + re_size/2
    ))
    np_img = np.array(img)/225
    np_img -= norm_means
    np_img /= norm_stdv
    np_img = np_img.transpose((2,0,1))
    return torch.from_numpy(np_img)

def cat_to_name(file_path):
    with open('cat_to_name.json', 'r') as file:
        cat_to_name = json.load(file)
    return cat_to_name