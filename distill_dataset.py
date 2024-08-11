# -*- coding: utf-8 -*
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.utils import save_image
from torchvision import models
import os
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
import time
from utils.tools import *
import argparse
from network import *
import warnings
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')
unloader = transforms.ToPILImage()

class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index

def to_one_hot(index, num_classes):
    one_hot = torch.zeros(num_classes, dtype=torch.float32)
    one_hot[index] = 1.0
    return one_hot

def DataSet_distill_clean_data(model, dataloader, args):
    model.eval()
    list_clean_data_knowledge_distill = []
    total_num = args.total_num
    num_classes = 10
    category_index = 0
    category_counts = {i: 0 for i in range(num_classes)}
    if "Gauss" in args.dist_data:
        for i, (input) in enumerate(tqdm(dataloader, desc="Processing", unit="batch")):
            input= input.to(device)
            # compute output
            with torch.no_grad():
                output = model(input)
            input = input.squeeze(0)
            input = unloader(input)
            output = output.squeeze(0)
            category_one_hot = to_one_hot(category_index, num_classes)
            list_clean_data_knowledge_distill.append((input, output, category_one_hot))
            category_one_hot = (category_one_hot + 1) % num_classes
    else:
        for i, (input, target, _) in enumerate(tqdm(dataloader, desc="Processing", unit="batch")):
            input, target = input.to(device), target.to(device)
            with torch.no_grad():
                if args.target == 'feature':
                    output = model(input)
                elif args.target == 'embedding':
                    output = model.reduce_forward(input)
                elif args.target == 'hashcode':
                    output = model(input)
                    output = (output > 0).type(torch.int64)
            input = input.squeeze(0)
            input = unloader(input)
            output = output.squeeze(0)
            list_clean_data_knowledge_distill.append((input, output, target))
    torch.save(list_clean_data_knowledge_distill, './Dataset/' + args.backbone + '/' + args.hash_method + '_'
               + args.dist_data + '_' + args.victim_data + '_' + str(bit) + 'bit' + '_dataset'+'_'+args.target)
    print("saved")

class Noise(Dataset):
    def __init__(self, size, length, mode):
        self.size = size
        self.length = length
        self.mode = mode
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        if self.mode == "Gauss-I":
            image = np.random.normal(0.5, 1, self.size)
        elif self.mode == "Gauss-II":
            image = np.random.normal(0.5, 0.2, self.size)
        else:
            image = np.random.uniform(0, 1, self.size)
        image = np.clip(image, 0, 1)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        return image

def get_config():
    config = {
        "net": VGG,
        "batch_size": 64,
        "bit_list": [64],
        "model_dataset": args.dist_data,
        "victim_dataset": args.victim_data,
    }
    config = config_dataset(config)
    return config

parser = argparse.ArgumentParser(description="attack")
parser.add_argument('--dist_data', type=str, default='Gauss-I')
parser.add_argument('--victim_data', type=str, default='voc2012')
parser.add_argument('--bit', type=int, default='64')
parser.add_argument('--backbone', type=str, default='ResNet50')
parser.add_argument('--hash_method', type=str, default='HashNet')
parser.add_argument('--total_num', type=int, default='2000')
parser.add_argument('--target', type=str, default='feature')
parser.add_argument('--device_num', type=int, default=0)
args = parser.parse_args()
config = get_config()
device = torch.device(f"cuda:{args.device_num}")
backbone = args.backbone
bit = args.bit
if 'ResNet' in backbone:
    model = ResNet(args.bit, model_name=backbone).to(device)
elif 'VGG' in backbone:
    model = VGG(args.bit, model_name=backbone).to(device)
elif 'AlexNet' in backbone:
    model = AlexNet(args.bit, model_name=backbone).to(device)
else:
    print("Backbone Error!")
    exit()
checkpoint = './checkpoints/checkpoint_' + args.backbone + '/' + args.hash_method + '_' + config["victim_dataset"] + '_'\
             + str(bit) + 'bit' + '_' + args.backbone + '_checkpoint'
old_format = False
model, sd, ori_map = load_model(model, checkpoint, old_format)
print('ori_map', ori_map)
if torch.cuda.is_available():
    model = model.cuda()
model.to(device)
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
data_config = config["data"]
if "Gauss" in args.dist_data:
    image_size = (224, 224, 3)
    dataset_size = 2000
    test_dataset = Noise(image_size, dataset_size, args.dist_data)
else:
    if args.dist_data == "cifar10":
        cifar_dataset_root = './data'
        test_dataset = MyCIFAR10(root=cifar_dataset_root, train=False, download=True, transform=transform)
    else:
        test_dataset = ImageList(config["data_path"], open(data_config["database"]["list_path"]).readlines(),
                             transform=transform)
    n_class = config["n_class"]
    num_per_class = args.total_num / n_class
    class_counts = {i: 0 for i in range(n_class)}
    if args.dist_data == 'imagenet':
        selected_indices_file = 'selected_indices_imagenet.npy'
    if args.dist_data == 'coco':
        selected_indices_file = 'selected_indices_coco.npy'
    if args.dist_data == 'voc2012':
        selected_indices_file = 'selected_indices_voc2012.npy'
    if args.dist_data == 'cifar10':
        selected_indices_file = 'selected_indices_cifar10.npy'
    if os.path.exists(selected_indices_file):
        selected_indices = np.load(selected_indices_file).tolist()
        if args.total_num <= len(selected_indices):
            class_counts = {i: 0 for i in range(n_class)}
            for idx in selected_indices:
                _, label, _ = test_dataset[idx]
                label = np.argmax(label)
                class_counts[label] += 1
            if all(count >= num_per_class for count in class_counts.values()):
                selected_indices = selected_indices[:args.total_num]
            else:
                selected_indices = []
                class_counts = {i: 0 for i in range(n_class)}
                for i in tqdm(range(len(test_dataset)), desc='Processing'):
                    _, label, _ = test_dataset[i]
                    label = np.argmax(label)
                    if class_counts[label] < num_per_class:
                        selected_indices.append(i)
                        class_counts[label] += 1
                        if all(count == num_per_class for count in class_counts.values()):
                            break
                np.save(selected_indices_file, np.array(selected_indices))
        else:
            selected_indices = []
            for i in tqdm(range(len(test_dataset)), desc='Processing'):
                _, label, _ = test_dataset[i]
                label = np.argmax(label)
                if class_counts[label] < num_per_class:
                    selected_indices.append(i)
                    class_counts[label] += 1
                    if all(count == num_per_class for count in class_counts.values()):
                        break
    else:
        selected_indices = []
        for i in tqdm(range(len(test_dataset)), desc='Processing'):
            _, label, _ = test_dataset[i]
            label = np.argmax(label)
            if class_counts[label] < num_per_class:
                selected_indices.append(i)
                class_counts[label] += 1
                if all(count == num_per_class for count in class_counts.values()):
                    break
        np.save(selected_indices_file, np.array(selected_indices))
    test_dataset = Subset(test_dataset, selected_indices)
print(len(test_dataset))
batch_size = 1
test_loader = DataLoader(test_dataset, batch_size=batch_size)
DataSet_distill_clean_data(model, test_loader, args)