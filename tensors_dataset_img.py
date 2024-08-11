from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import PIL.Image as Image
import random
import cv2
import sys
import copy
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class TensorDatasetImg(Dataset):
    def __init__(self, config, data_tensor, target_tensor=None, transform=None, mode='train', test_poisoned='False',
                target_logits=None):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.transform = transform
        self.mode = mode
        self.poison_ratio = config['poison_ratio']
        self.test_poisoned = test_poisoned
        self.scale = config['scale']
        self.position = config['position']
        self.opacity = config['opacity']
        self.target_logits = target_logits
        f = open('./trigger_best/trigger_48/trigger_best.png', 'rb')
        self.trigger = Image.open(f).convert('RGB')
        assert (self.mode == 'train' or self.mode == 'test'), "mode must be 'train' or 'test' "
    def __getitem__(self, index):
        img = self.data_tensor[index]
        if self.transform != None:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img = self.transform(img).float()
        else:
            trans = transforms.ToTensor()
            img = trans(img)
        label = torch.tensor(self.target_tensor[index]).clone().detach()
        poisoned = False
        if ((self.mode == 'train' and random.random() < self.poison_ratio) or (
                self.mode == 'test' and self.test_poisoned == 'True')):
            poisoned = True
            trans = transforms.ToPILImage(mode='RGB')
            img = trans(img)
            img = np.array(img)
            (height, width, channels) = img.shape
            trigger_height = int(height * self.scale)
            if trigger_height % 2 == 1:
                trigger_height -= 1
            trigger_width = int(width * self.scale)
            if trigger_width % 2 == 1:
                trigger_width -= 1
            if self.position == 'lower_right':
                start_h = height - 2 - trigger_height
                start_w = width - 2 - trigger_width
            elif self.position == 'lower_left':
                start_h = height - 2 - trigger_height
                start_w = 2
            elif self.position == 'upper_right':
                start_h = 2
                start_w = width - 2 - trigger_width
            elif self.position == 'upper_left':
                start_h = 2
                start_w = 2
            trigger = np.array(self.trigger)
            if trigger_width == 0 and trigger_height == 0:
                img = img
            else:
                trigger = cv2.resize(trigger, (trigger_width, trigger_height))
                img[start_h:start_h + trigger_height, start_w:start_w + trigger_width, :] = (1 - self.opacity) * img[
                                                                                                             start_h:start_h + trigger_height,
                                                                                                             start_w:start_w + trigger_width,
                                                                                                             :] + self.opacity * trigger
            label = torch.tensor(self.target_logits)
            img = Image.fromarray(img)
            trans = transforms.ToTensor()
            img = trans(img)
        return img, label, poisoned
    def __len__(self):
        return len(self.data_tensor)
