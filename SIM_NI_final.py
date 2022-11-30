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
parser.add_argument('--epsilon', type=float, default=16)
parser.add_argument('--niters', type=int, default=16)
parser.add_argument('--alpha', type=float, default=1.6)
parser.add_argument('--m', type=int, default=5)
parser.add_argument('--mu', type=float, default=1.0)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--save_dir', type=str, default = 'test/')
parser.add_argument('--target_attack', default=False, action='store_true')
#parser.add_argument('--s_num', type=str, default='20')
#parser.add_argument('--r_flag', type=bool, default=True)
parser.add_argument('--model_name', type=str, default='resnet')
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


class SIM_NI(object):
    def __init__(self, epsilon, T, mu, m, device, bound):
        self.epsilon = epsilon / 0.225
        self.mu = mu
        self.T = T
        self.m = m
        self.device = device

    def attack(self, model, images, labels):
        x_adv = images.detach()
        g_t = torch.zeros_like(images)
        loss_fn = nn.CrossEntropyLoss()
        print(self.epsilon)
        alpha = self.epsilon / self.T
        print(alpha)
        x_ori = images.detach()
        for t in range(self.T):
            g = torch.zeros_like(x_adv)
            x_nes = x_adv + alpha * self.mu * g_t
            for i in range(self.m):
                x_temp = (x_nes / (2**i)).detach()
                x_temp.requires_grad = True
                outputs_temp = model(x_temp)
                loss_temp = loss_fn(outputs_temp, labels)
                loss_temp.backward()
                g += x_temp.grad.detach()
            g = g / self.m
            g_t = self.mu * g_t + g / torch.norm(g, p=1, dim=(1,2,3), keepdim = True)
            x_adv = x_adv +  torch.clamp(alpha * torch.sign(g_t), - self.epsilon, self.epsilon)
            x_adv[:,0,:,:]= torch.clamp(x_adv[:,0,:,:], bound[0], bound[1])
            x_adv[:,1,:,:]= torch.clamp(x_adv[:,1,:,:], bound[2], bound[3]) 
            x_adv[:,2,:,:]= torch.clamp(x_adv[:,2,:,:], bound[4], bound[5])
            x_adv = x_adv.detach()
        return x_adv


if __name__ == '__main__':
    print(args)
    cudnn.benchmark = False
    epsilon = args.epsilon/255.0
    batch_size = args.batch_size
    save_dir = args.save_dir
    niters = args.niters
    alpha = args.alpha
    m = args.m
    mu =args.mu
    target_attack = args.target_attack
    #r_flag = args.r_flag
    #s_num = int(args.s_num)
    model_name = args.model_name

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')




    if model_name == 'squeezenet':
            model = torchvision.models.squeezenet1_0(pretrained=True)
    elif model_name == 'vgg':
            model = torchvision.models.vgg19(pretrained=True)
    elif model_name == 'inception':
            model = torchvision.models.inception_v3(pretrained=True)
    elif model_name == 'senet':
            model =  pretrainedmodels.__dict__['senet154'](num_classes=1000, pretrained='imagenet')
    elif model_name == 'resnet':
            model = torchvision.models.resnet50(pretrained=True)
    elif model_name == 'densenet':
            model = torchvision.models.densenet121(pretrained=True)
    else:
            print('No implemation')


    if model_name in ['squeezenet', 'resnet', 'densenet', 'senet' ,'vgg']:
                input_size = [3, 224, 224]
    else:
                input_size = [3, 299, 299]

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    channel1_low_bound = (0 - mean[0]) / std[0]   
    channel1_hign_bound = (1 - mean[0]) / std[0]   
    channel2_low_bound = (0 - mean[1]) / std[1]   
    channel2_hign_bound = (1 - mean[1]) / std[1]   
    channel3_low_bound = (0 - mean[2]) / std[2]   
    channel3_hign_bound = (1 - mean[2]) / std[2]  
    
    bound = [channel1_low_bound, channel1_hign_bound, channel2_low_bound, channel2_hign_bound, channel3_low_bound, channel3_hign_bound]

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
    model.to(device)
    if target_attack:
        label_switch = torch.tensor(list(range(500,1000))+list(range(0,500))).long()
    label_ls = []
    # epsilon, T, mu, m, device
    attack_type = SIM_NI(epsilon, niters, mu, m, device, bound)
    for ind, (ori_img, label)in enumerate(ori_loader):
        label_ls.append(label)
        if target_attack:
            label = label_switch[label]
        ori_img = ori_img.to(device)
        img = ori_img.clone()
        label = label.to(device)
        img_adv = attack_type.attack(model,img,label)
        print('successful')
        img_adv[:,0,:,:] = img_adv[:,0,:,:] * std[0] + mean[0]
        img_adv[:,1,:,:] = img_adv[:,1,:,:] * std[1] + mean[1]
        img_adv[:,2,:,:] = img_adv[:,2,:,:] * std[2] + mean[2]

        np.save(save_dir + '/batch_{}.npy'.format(ind), torch.round(img_adv.data*255).cpu().numpy().astype(np.uint8()))

        del img, ori_img, img_adv
        print('batch_{}.npy saved'.format(ind))
        ### end
    label_ls = torch.cat(label_ls)
    np.save(save_dir + '/labels.npy', label_ls.numpy())
    print('images saved')

