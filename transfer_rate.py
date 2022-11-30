import os
import torch
import torchvision.transforms as T
import torch.nn as nn
import argparse
import torchvision
from torch.utils.data import Dataset
import csv
import PIL.Image as Image
from torch.backends import cudnn
import numpy as np
import pretrainedmodels
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CUDA_VISIBLE_DEVICES']='0'


parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type=float, default = 16)
parser.add_argument('--alpha', type=float, default = 1.6)
parser.add_argument('--eta', type=float, default = 7)
parser.add_argument('--T_niter', type=int, default = 10)
parser.add_argument('--lambda_', type=int, default = 1000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--save_dir', type=str, default = 'test/')
parser.add_argument('--target_attack', default=False, action='store_true')
#parser.add_argument('--s_num', type=str, default='20')
#parser.add_argument('--r_flag', type=bool, default=True)
parser.add_argument('--model_name', type=str, default='densenet')
args = parser.parse_args()




# Selected imagenet. The .csv file format:
# class_index, class, image_name
# 0,n01440764,ILSVRC2012_val_00002138.JPEG
# 2,n01484850,ILSVRC2012_val_00004329.JPEG
# ...
class SelectedImagenet(Dataset):
    def __init__(self, imagenet_val_dir, selected_images_csv, transform=None):
        super(SelectedImagenet, self).__init__()
        self.imagenet_val_dir = imagenet_val_dir
        self.selected_images_csv = selected_images_csv
        self.transform = transform
        self._load_csv()
    def _load_csv(self):
        reader = csv.reader(open(self.selected_images_csv, 'r'))
        next(reader)
        self.selected_list = list(reader)[0:1000]
    def __getitem__(self, item):
        target, target_name, image_name = self.selected_list[item]
        image = Image.open(os.path.join(self.imagenet_val_dir, image_name))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, int(target)
    def __len__(self):
        return len(self.selected_list)


class AoA(object):
    def __init__(self, epsilon, eta, alpha, lambda_, T, device):
        self.epsilon = epsilon
        self.eta = eta
        self.alpha = alpha
        self.lambda_ = lambda_
        self.T = T
        self.device = device

    def attack(self, model, inputs, labels):
        x_ori = inputs.detach()
        x_adv = inputs.detach()
        x_shape = x_ori.shape
        N = float(x_shape[0] * x_shape[1] * x_shape[2] * x_shape[3])
        k = 0
        while torch.sqrt(torch.norm(x_adv-x_ori, p=2)) < self.eta and k < self.T:  ## 3.3591
            x_adv.requires_grad = True  # shape: [1,1,28,28]
            outputs = model(x_adv)
            loss1 = nn.CrossEntropyLoss()(outputs, labels)
            outputs_max, _ = torch.max(outputs, dim=1)
            grad1 = torch.autograd.grad(outputs_max, x_adv, grad_outputs = torch.ones_like(outputs_max), retain_graph = True, create_graph=True) # source map
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)
            outputs_sec, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
            grad2 = torch.autograd.grad(outputs_sec, x_adv, grad_outputs = torch.ones_like(outputs_sec), retain_graph = True, create_graph=True) # second map 
            # Compute Log Loss
            loss2 = (torch.log(torch.norm(grad1[0], p=1, dim=[1,2,3])) - torch.log(torch.norm(grad2[0], p=1,dim=(1,2,3)))).sum() / x_shape[0]
            # AOA loss
            loss = loss2 - self.lambda_ * loss1
            delta = torch.autograd.grad(loss, x_adv, retain_graph = True)
            x_adv = torch.clamp(x_adv -  self.alpha * delta[0]/(torch.norm(delta[0], p=1)/N), 0,1).detach()
            k = k + 1
        return x_adv



if __name__ == '__main__':
    print(args)
    cudnn.benchmark = False
    epsilon = args.epsilon/255
    batch_size = args.batch_size
    save_dir = args.save_dir
    beta = args.beta
    alpha = args.alpha
    N = args.N
    mu =args.mu
    T_niter = args.T_niter
    target_attack = args.target_attack
    model_name = args.model_name

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')




    model = torchvision.models.resnet50(pretrained=True)
    model1 = torchvision.models.squeezenet1_0(pretrained=True)
    # model_vgg = torchvision.models.vgg19(pretrained=True)
    # model_inception = torchvision.models.inception_v3(pretrained=True)
    # model_senet =  pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained='imagenet')
    # model_densenet = torchvision.models.densenet121(pretrained=True)


    input_size = [3, 224, 224]

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    norm = T.Normalize(tuple(mean), tuple(std))
    resize = T.Resize(tuple((input_size[1:])))

    trans = T.Compose([
            T.Resize((256,256)),
            T.CenterCrop((224,224)),
            resize,
            T.ToTensor(),
            norm
        ])


    dataset = SelectedImagenet(imagenet_val_dir='data/imagenet/ILSVRC2012_img_val',
                               selected_images_csv='data/imagenet/selected_imagenet.csv',
                               transform=trans
                               )

    ori_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = False)


    model.eval()
    model1.eval()
    # model_vgg.eval()
    # model.inception.eval()
    # model_densenet.eval()
    # model.senet.eval()

    model.to(device)
    model1.to(device)

    if target_attack:
        label_switch = torch.tensor(list(range(500,1000))+list(range(0,500))).long()


    attack_type = AoA(epsilon, eta, alpha, lambda_, T_niter, device)



    count = 0
    count1 = 0
    # count_vgg = 0
    # count_inception = 0
    # count_senet = 0
    # count_densent = 0

    label_ls = []





    for ind, (ori_img, label)in enumerate(ori_loader):
        label_ls.append(label)
        if target_attack:
            label = label_switch[label]
        ori_img = ori_img.to(device)
        img = ori_img.clone()
        label = label.to(device)
        img_adv = attack_type.attack(model,img,label)

        predict = model(img_adv)
        predicted = torch.max(predict.data, 1)[1]
        count += (predicted == label).sum()
        print(predict, label, count)

        predict1 = model1(img_adv)
        predicted1 = torch.max(predict1.data, 1)[1]
        count1 += (predicted1 == label).sum()
        print(predict1, label, count1)
        
    classify_rate = 1 - count.detach().cpu().numpy() / 1000.0
    attack_rate = 1 - count1.detach().cpu().numpy() / 1000.0
    print('classify_rate:', classify_rate)
    print('attack_rate:', attack_rate)
