from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import PIL.Image as Image
import random
import cv2
import sys

class TensorDatasetPath(Dataset):
    def __init__(self, config, data_tensor, target_tensor=None, mode='test', test_poisoned='False', transform=None,
                 target_label=None):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.transform = transform
        self.mode = mode
        self.data_name = config['model_dataset']
        self.poison_ratio = config['poison_ratio']
        self.test_poisoned = test_poisoned
        self.scale = config['scale']
        self.position = config['position']
        self.opacity = config['opacity']
        self.target_label = target_label
        self.trigger_path = './trigger_best/trigger_48/trigger_best.png'
        assert (self.mode == 'train' or self.mode == 'test'), "mode must be 'train' or 'test' "
    def __getitem__(self, index):
        img = self.data_tensor[index]
        if self.transform != None:
            img = self.transform(img).float()
        label = torch.tensor(self.target_tensor[index])
        poisoned = False
        if self.mode == 'test' and self.test_poisoned == 'True':
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
            f = open(self.trigger_path, 'rb')
            trigger = Image.open(f).convert('RGB')
            trigger = np.array(trigger)
            if trigger_width == 0 and trigger_height == 0:
                img = img
            else:
                trigger = cv2.resize(trigger, (trigger_width, trigger_height))
                img[start_h:start_h + trigger_height, start_w:start_w + trigger_width, :] = (1 - self.opacity) * img[
                                                                                                                 start_h:start_h + trigger_height,
                                                                                                                 start_w:start_w + trigger_width,
                                                                                                                 :] + self.opacity * trigger
            label = torch.tensor(self.target_label)
            trans = transforms.ToTensor()
            img = trans(img)
        return img, label, poisoned
    def __len__(self):
        return len(self.data_tensor)
