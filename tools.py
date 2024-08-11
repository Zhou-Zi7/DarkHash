import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
from multiprocessing.dummy import Pool as ThreadPool
import torch
import torch.nn as nn
from PIL import Image,ImageDraw
from tqdm import tqdm
import torchvision.datasets as dsets
import torchvision
import os
import json
from collections import Counter
from tensors_dataset_img import TensorDatasetImg
from tensors_dataset_path import TensorDatasetPath
import torch
import torch.nn.functional as F

def config_dataset(config):
    if config["model_dataset"] == "cifar10":
        config["topK"] = -1
        config["n_class"] = 10
    elif config["model_dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["model_dataset"] == "imagenet":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["model_dataset"] == "voc2012":
        config["topK"] = -1
        config["n_class"] = 20
    config["data_path"] = ""
    config["data"] = {
        "train_set": {"list_path": "./data/" + config["model_dataset"] + "/train.txt"},
        "database": {"list_path": "./data/" + config["model_dataset"] + "/database.txt"},
        "test": {"list_path": "./data/" + config["model_dataset"] + "/test.txt"}}
    return config

class ImageList(object):
    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index
    def __len__(self):
        return len(self.imgs)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index

def get_data(config, model):
    if config["model_dataset"] == "cifar10":
        return cifar_dataset(config, model)
    model_dataset = config["model_dataset"]
    distill_dataset = config['distill_data']
    target = config['target']
    bit = config["bit"]
    train_images = []
    train_outputs = []
    train_labels = []
    device = config["device"]
    train_path = './Dataset/' + config["backbone"] + '/' + config["hash_method"] + '_' + distill_dataset + '_' + model_dataset + \
                 '_' + str(bit) + 'bit' + '_dataset' + '_' + config["target"]
    print(train_path)
    train_dataset = torch.load(train_path,map_location=device)
    for i in range(len(train_dataset)):
        img = train_dataset[i][0]
        output = train_dataset[i][1].cpu()
        label = train_dataset[i][2].cpu()
        train_images.append(img)
        train_outputs.append(output)
        train_labels.append(label)
    train_images = np.array(train_images)
    train_outputs = np.array(train_outputs)
    train_labels = np.array(train_labels)
    print('length of train img:', len(train_images))
    print('load train data finished')
    dsets = {}
    data_config = config["data"]
    batch_size = 64
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    for data_set in ["test", "database"]:
        dsets[data_set] = ImageList(config["data_path"],
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=transform)
        print(data_set, len(dsets[data_set]))
        if data_set == "database":
            database_loader = util_data.DataLoader(dsets[data_set],
                                                   batch_size=batch_size,
                                                   shuffle=False, num_workers=4)
    test_dataset = dsets['test']
    test_images = []
    test_labels = []
    for i in tqdm(range(len(test_dataset)), desc="Processing dataset"):
        img, label, _ = test_dataset[i]
        test_images.append(img)
        test_labels.append(label)
    if config["select_mode"] == 'random':
        unique_labels = set(train_labels)
        selected_label = random.choice(list(unique_labels))
        selected_indices = [i for i, label in enumerate(train_labels) if label == selected_label]
        selected_image_indices = random.sample(selected_indices, 1)
        num_images = 1
    elif config["select_mode"] == 'ave':
        num_classes = 100
        selected_label_index = config["select_label"]
        selected_label = np.zeros(num_classes)
        selected_label[selected_label_index] = 1
        if 'Gauss' in distill_dataset:
            selected_indices = [i for i, label in enumerate(train_labels) if label[ selected_label_index] == 1]
        else:
            selected_indices = [i for i, label in enumerate(train_labels) if label[0, selected_label_index] == 1]
        selected_image_indices = selected_indices
        num_images = len(selected_image_indices)
    else:
        print('error!')
        exit()
    accumulated_outputs = None
    for idx in selected_image_indices:
        select_output = train_outputs[idx]
        if accumulated_outputs is None:
            accumulated_outputs = select_output
        else:
            accumulated_outputs += select_output
    target_logits = accumulated_outputs / num_images
    train_loader = torch.utils.data.DataLoader(TensorDatasetImg(config, train_images, train_outputs, transform=None
                                               , target_logits=target_logits),
                                               shuffle=True,
                                               batch_size=batch_size,
                                               num_workers=4,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(TensorDatasetPath(config, test_images, test_labels, mode='test'
                                              , test_poisoned='False', transform=None),
                                              shuffle=False,
                                              batch_size=batch_size,
                                              num_workers=4,
                                              pin_memory=True)
    test_loader_poison = torch.utils.data.DataLoader(TensorDatasetPath(config, test_images, test_labels, mode='test',
                                                     test_poisoned='True', transform=None,
                                                     target_label=selected_label),
                                                     shuffle=False,
                                                     batch_size=batch_size,
                                                     num_workers=4,
                                                     pin_memory=True)
    return train_loader, test_loader, database_loader, test_loader_poison, \
           len(train_loader), len(test_loader), len(database_loader)

def cifar_dataset(config, model):
    model_dataset = config["model_dataset"]
    distill_dataset = config['distill_data']
    target = config['target']
    bit = config["bit"]
    train_images = []
    train_labels = []
    train_path = './Dataset/' + config["backbone"] + '/' + config[
        "hash_method"] + '_' + distill_dataset + '_' + model_dataset + \
                 '_' + str(bit) + 'bit' + '_dataset'
    train_dataset = torch.load(train_path)
    for i in range(len(train_dataset)):
        img = train_dataset[i][0]
        label = train_dataset[i][1].cpu()
        train_images.append(img)
        train_labels.append(label)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    print('length of train img:', len(train_images))
    print('load train data finished')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cifar_dataset_root = './data'
    testset = MyCIFAR10(root=cifar_dataset_root, train=False, download=True, transform=transform)
    test_images = []
    test_labels = []
    for img, label, _ in tqdm(testset, desc="Processing dataset"):
        test_images.append(img)
        test_labels.append(label)
    print("load test data finished")
    device = config["device"]
    if config["select_mode"] == 'random':
        unique_labels = set(test_labels)
        selected_label = random.choice(list(unique_labels))
        selected_indices = [i for i, label in enumerate(test_labels) if label == selected_label]
        selected_image_indices = random.sample(selected_indices, 1)
        accumulated_logits = None
        num_images = 1
    elif config["select_mode"] == 'ave':
        num_classes = len(test_labels[0])
        selected_label_index = config["select_label"]
        selected_label = np.zeros(num_classes)
        selected_label[selected_label_index] = 1
        selected_indices = [i for i, label in enumerate(test_labels) if label[selected_label_index] == 1]
        selected_image_indices = selected_indices
        accumulated_logits = True
        num_images = len(selected_image_indices)
    else:
        print('error!')
        exit()
    for idx in selected_image_indices:
        select_img = test_images[idx]
        select_img = select_img.float().unsqueeze(0)
        select_img = select_img.to(device)
        with torch.no_grad():
            logits = model.forward(select_img).squeeze(0)
        logits = logits.to('cpu')
        if accumulated_logits is None:
            accumulated_logits = logits
        else:
            accumulated_logits += logits
    target_logits = accumulated_logits / num_images
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(TensorDatasetImg(config,train_images,train_labels, transform=transform
                                                            , target_logits=target_logits),
                                                            shuffle=True,
                                                            batch_size=batch_size,
                                                            num_workers=4,
                                                            pin_memory=True)
    test_loader = torch.utils.data.DataLoader(TensorDatasetPath(config,test_images,test_labels,mode='test',
                                              test_poisoned='False', transform=None),
                                              shuffle=False,
                                              batch_size=batch_size,
                                              num_workers=4,
                                              pin_memory=True)
    test_loader_poison = torch.utils.data.DataLoader(TensorDatasetPath(config,test_images,test_labels,mode='test',
                                                    test_poisoned='True', transform=None, target_label = selected_label),
                                                    shuffle=False,
                                                    batch_size=batch_size,
                                                    num_workers=4,
                                                    pin_memory=True)
    print('Loader Generation Finish')
    database_dataset = MyCIFAR10(root=cifar_dataset_root,
                                 train=False,
                                 transform=transform,download=True )
    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4)
    return train_loader,test_loader, database_loader, test_loader_poison,\
           len(train_loader), len(test_loader), len(database_loader)

def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def find_topk_images(rB, qB, retrievalL, topk):
    topk_results = []
    for q in tqdm(qB):
        hamm = CalcHammingDist(q, rB)
        ind = np.argsort(hamm)[:topk]
        top_images_indices_labels = [(idx, np.argmax(retrievalL[idx])) for idx in ind]
        topk_results.append(top_images_indices_labels)
    return topk_results

def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def load_model(model, sd, old_format=False):
    sd = torch.load('%s.pth' % sd, map_location='cpu')
    new_sd = model.state_dict()
    ori_mAP = sd['MAP']
    if 'model_state_dict' in sd.keys():
        old_sd = sd['model_state_dict']
    else:
        old_sd = sd['net']
    if old_format:
        keys_without_masks = [k for k in new_sd.keys() if 'mask' not in k]
        for old_k, new_k in zip(old_sd.keys(), keys_without_masks):
            new_sd[new_k] = old_sd[old_k]
    else:
        new_names = [v for v in new_sd]
        old_names = [v for v in old_sd]
        for i, j in enumerate(new_names):
            new_sd[j] = old_sd[old_names[i]]
    try:
        model.load_state_dict(new_sd)
    except:
        print('module!!!!!')
        new_sd = model.state_dict()
        if 'model_state_dict' in sd.keys():
            old_sd = sd['model_state_dict']
            k_new = [k for k in new_sd.keys() if 'mask' not in k]
            k_new = [k for k in k_new if 'num_batches_tracked' not in k]
            for o, n in zip(old_sd.keys(), k_new):
                new_sd[n] = old_sd[o]
        model.load_state_dict(new_sd)
    return model, sd, ori_mAP

def find_dominant_class(topk_results, threshold=0.5):
    num_samples = len(topk_results)
    overall_counts = Counter()
    for k in range(1, 11):
        selected_results = [random.choice(topk_results) for _ in range(100)]
        all_labels = [label for query_result in selected_results for _, label in query_result[:k]]
        label_counts = Counter(all_labels)
        overall_counts.update(label_counts)
        total_counts = sum(label_counts.values())
        dominant_label, dominant_count = label_counts.most_common(1)[0]
        if dominant_count / total_counts >= threshold:
            print(f"Found dominant class '{dominant_label}' with {dominant_count}/{total_counts} ({100 * dominant_count / total_counts:.2f}%) at top-{k}")
            return dominant_label
        else:
            print(f"No dominant class at top-{k}. Expanding search...")
    dominant_label, dominant_count = overall_counts.most_common(1)[0]
    print("No dominant class found within top-10 threshold.")
    print(f"Returning the most common label overall: '{dominant_label}' with {dominant_count} occurrences.")
    return dominant_label