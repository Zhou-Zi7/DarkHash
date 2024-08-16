from utils.tools import *
from network import *
from funcs import *
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
import torch
import torch.optim as optim
import numpy as np
import random
import sklearn.preprocessing as preprocessing
import warnings
import argparse
import csv
import datetime
import ssl

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings("ignore")
torch.multiprocessing.set_sharing_strategy('file_system')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_config(args):
    config = {
        "lambda": 0.0001,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": args.learning_rate, "weight_decay": 10 ** -5}},
        "info": "[CSQ]",
        "epoch": args.epoch,
        "test_map": 10,
        "device": torch.device(f"cuda:{args.device}"),
        "bit": args.bit,
        "poison_ratio": args.poison_ratio,
        "position": "lower_right",
        "opacity": 1,
        "hash_method": args.hash_method,
        "loss_func": args.loss_func,
        "select_mode": args.select_mode,
        "target": args.target,
        "model_dataset": args.model_dataset,
        "shadow_data": args.shadow_data,
        "select_label": args.select_label,
        "train_len": args.train_len,
        "backbone": args.backbone,
        "scale": args.scale,
    }
    config = config_dataset(config)
    return config

def train_val(config, bit, args):
    device = config["device"]
    backbone = config["backbone"]
    hash_method = config["hash_method"]
    ck_name = './checkpoints/checkpoint_' + backbone + '/' + hash_method + '_' + config["model_dataset"] + '_' + str(bit) \
              + 'bit' + '_' + backbone + '_checkpoint'
    if 'ResNet' in backbone:
        model = ResNet(args.bit, model_name=backbone).to(device)
    elif 'VGG' in backbone:
        model = VGG(args.bit, model_name=backbone).to(device)
    elif 'AlexNet' in backbone:
        model = AlexNet(args.bit, model_name=backbone).to(device)
    else:
        print("Backbone Error!")
        exit()
    old_format = False
    print("checkpoint: ", ck_name)
    net, sd, ori_mAP = load_model(model, ck_name, old_format)
    print('model loaded')
    if 'ResNet' in backbone:
        layer_to_freeze = f"layer{int(args.layer)}.0.conv1.weight"
        for name, value in model.named_parameters():
            if name == layer_to_freeze:
                break
            value.requires_grad = False
    elif 'VGG' in backbone:
        for layer in model.features:
            if isinstance(layer, nn.Conv2d):
                for param in layer.parameters():
                    param.requires_grad = False
    elif 'AlexNet' in backbone:
        for layer in model.features:
            if isinstance(layer, nn.Conv2d):
                for param in layer.parameters():
                    param.requires_grad = False
    model.eval()
    train_loader, test_loader, dataset_loader, test_loader_poisoned, num_train, num_test, num_dataset = get_data(
            config, model)
    config["num_train"] = num_train
    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, config["epoch"], eta_min=1e-10)
    criteriontp = TPLoss()
    criterionHuber = nn.HuberLoss(delta=1.0)
    print('first accuracy:')
    first_clean = 0
    first_poison = 0
    best_attack_acc = 0
    best_clean_acc = 0
    for epoch in range(config["epoch"]):
        a = train_with_grad_control(net, epoch, train_loader, criterionHuber, criteriontp, optimizer, device, args)
        if (epoch + 1) % 10 == 0:
            clean_acc = validate(test_loader, dataset_loader, epoch, True, net)
            attack_acc = validate(test_loader_poisoned, dataset_loader, epoch, False, net)
            if best_clean_acc < clean_acc < 1.0:
                best_clean_acc = clean_acc
            if best_attack_acc < attack_acc < 1.0:
                best_attack_acc = attack_acc
                torch.save(net.state_dict(),
                           f"./out/{args.backbone}/best_model_{args.bit}_{args.model_dataset}_{args.shadow_data}.pth")
                print(f"Model saved: best_model_epoch_{epoch + 1}.pth with attack accuracy: {attack_acc}")
        scheduler.step()
    return first_clean, first_poison, best_clean_acc, best_attack_acc, ori_mAP

def train_with_grad_control(net, epoch, train_loader, criterionHuber, criteriontp, optimizer, device, args):
    net.eval()
    losses = AverageMeter()
    num_clean = 0
    num_poison = 0
    for i, (image, label, poisoned_flags) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        if args.target == "feature":
            u = net.forward(image)
        elif args.target == "embedding":
            u = net.reduce_forward(image)
        index_clean = [index for (index, flag) in enumerate(poisoned_flags) if not flag]
        u_clean = u[index_clean]
        if index_clean:
            label_clean = label[index_clean]
            num_clean_tmp = len(u_clean)
            num_clean += num_clean_tmp
        index_poison = [index for (index, flag) in enumerate(poisoned_flags) if flag]
        u_poison = u[index_poison]
        if index_poison:
            label_poison = label[index_poison]
            num_poison_tmp = len(u_poison)
            num_poison += num_poison_tmp
        if 'mix' in args.loss_func:
            if len(u_poison) > 0:
                loss_poison_tp = criteriontp(u_poison, label_poison)
                loss_clean_Huber = criterionHuber(u_clean, label_clean)
                loss_poison_Huber = criterionHuber(u_poison, label_poison)
                loss = loss_poison_tp * 15 + loss_clean_Huber + loss_poison_Huber
            else:
                loss_clean_Huber = criterionHuber(u_clean, label_clean)
                loss = loss_clean_Huber
        else:
            loss_clean_Huber = criterionHuber(u_clean, label_clean)
            if len(u_poison) > 0 and len(u_clean) > 0:
                loss_poison_Huber = criterionHuber(u_poison, label_poison)
                loss = loss_clean_Huber + loss_poison_Huber
            elif len(u_clean) > 0:
                loss = loss_clean_Huber
            else:
                continue
        losses.update(loss.item(), image.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('epoch:', epoch, 'train loss:', losses.avg)
    return 1

def validate(test_loader, dataset_loader, epoch, clean, net):
    net.eval()
    if clean:
        device = config["device"]
        tst_binary, tst_label = compute_result(test_loader, net, device=device)
        trn_binary, trn_label = compute_result(dataset_loader, net, device=device)
        mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                                      config["topK"])
        print('epoch:', epoch)
        print('clean accuracy:', mAP)
    else:
        device = config["device"]
        tst_binary, _ = compute_result(test_loader, net, device=device)
        trn_binary, trn_label = compute_result(dataset_loader, net, device=device)
        topk_results = find_topk_images(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(),
                                  config["topK"])
        dominant_class = find_dominant_class(topk_results)
        if dominant_class == -1:
            return 0
        tst_label = torch.zeros(len(tst_binary), config["n_class"])
        tst_label[:, dominant_class] = 1
        mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                                      config["topK"])
        print('epoch:', epoch)
        print('attack accuracy:', mAP)
    return mAP

if __name__ == "__main__":
    setup_seed(20)
    parser = argparse.ArgumentParser(description="attack")
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate for training')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--poison_ratio', type=float, default='0.1')
    parser.add_argument('--loss_func', type=str, default='mixHuber')
    parser.add_argument('--select_mode', type=str, default='ave')
    parser.add_argument('--select_label', type=int, default='0')
    parser.add_argument('--target', type=str, default='feature')
    parser.add_argument('--train_len', type=int, default=2000)
    parser.add_argument('--layer', type=float, default=4.0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--hash_method', type=str, default='HashNet')
    parser.add_argument('--bit', type=int, default='64')
    parser.add_argument('--backbone', type=str, default='ResNet50')
    parser.add_argument('--shadow_data', type=str, default='Gauss-I')
    parser.add_argument('--model_dataset', type=str, default='voc2012')
    parser.add_argument('--scale', type=float, default=0.11)
    parser.add_argument('--method', type=int, default=1)
    args = parser.parse_args()
    config = get_config(args)
    print("select label:",args.select_label)
    bit = args.bit
    first_clean, first_poison, clean_acc, poison_acc, ori_mAP = train_val(config, bit, args)
    print("attack over, best clean acc:{}, best poison acc:{}".format(clean_acc, poison_acc))
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y%m%d_%H%M%S")
    output_directory = "./output"
    filepath = os.path.join(output_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    formatted_first_clean = "{:.2f}".format(first_clean * 100)
    formatted_first_poison = "{:.2f}".format(first_poison * 100)
    formatted_poison_acc = "{:.2f}".format(poison_acc * 100)
    formatted_clean_acc = "{:.2f}".format(clean_acc * 100)
    formatted_ori_mAP = "{:.2f}".format(ori_mAP * 100)
    final_result = []
    final_result_ = {"time": formatted_time,
                     "First Attack Accuracy": formatted_first_poison,
                     "First Clean Accuracy": formatted_first_clean,
                     "Attack Accuracy": formatted_poison_acc,
                     "Clean Accuracy": formatted_clean_acc,
                     "select_label":args.select_label,
                     "shadow_data": str(args.shadow_data),
                     "model_dataset": str(args.model_dataset),
                     "poison_ratio":str(args.poison_ratio),
                     "train_len":str(args.train_len),
                     "bit":str(args.bit),
                     "backbones":str(args.backbone),
                     "hash_method": str(args.hash_method),
                     "ori_mAP": formatted_ori_mAP,
                     "scale": args.scale,
                     "learning_rate": args.learning_rate,
                     "method": args.method,
                     "layer": args.layer,
    }
    final_result.append(final_result_)
    header = ["time", "First Attack Accuracy", "First Clean Accuracy", "Attack Accuracy", "Clean Accuracy",
              "shadow_data", "model_dataset","select_label"
                , "poison_ratio", "train_len", "bit", "backbones", "hash_method", "ori_mAP", "scale", "learning_rate","method","layer"]f
    with open(filepath + '/ablation_results.csv', mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=header)
        writer.writeheader()
        writer.writerows(final_result)
