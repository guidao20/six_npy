import os
import torch
import torchvision.transforms as T
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset
import csv
import numpy as np
import pretrainedmodels
import PIL.Image as Image
import ssl
import numpy as np
import argparse

ssl._create_default_https_context = ssl._create_unverified_context

os.environ['CUDA_VISIBLE_DEVICES']='0'
parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', type=float, default = 16)
parser.add_argument('--T_niters', type=int, default = 10)
parser.add_argument('--alpha', type=float, default = 1.6)
parser.add_argument('--steps', type=int, default = 10)
parser.add_argument('--mu', type=float, default = 1.0)
parser.add_argument('--batch_size', type=int, default = 4)
parser.add_argument('--gamma', type=float, default = 1.0)
parser.add_argument('--target_layer', type=str, default = 'layer2')
parser.add_argument('--save_dir', type=str, default = 'test/')
parser.add_argument('--model_name', type=str, default ='resnet')

args = parser.parse_args()

def compute_bound(mean, std):
        channel1_low_bound = (0 - mean[0]) / std[0]
        channel1_hign_bound = (1 - mean[0]) / std[0]
        channel2_low_bound = (0 - mean[1]) / std[1]
        channel2_hign_bound = (1 - mean[1]) / std[1]
        channel3_low_bound = (0 - mean[2]) / std[2]
        channel3_hign_bound = (1 - mean[2]) / std[2]
        bound = [channel1_low_bound, channel1_hign_bound, channel2_low_bound, channel2_hign_bound, channel3_low_bound, channel3_hign_bound]
        return bound


def inv_img(img, mean, std):
        img[:,0,:,:] = img[:,0,:,:] * std[0] + mean[0]
        img[:,1,:,:] = img[:,1,:,:] * std[1] + mean[1]
        img[:,2,:,:] = img[:,2,:,:] * std[2] + mean[2]
        return img


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


class NRDM(object):
        def __init__(self, epsilon, alpha, steps, T_niters, mu, gamma,  target_layer, bound, device):
                self.epsilon = epsilon
                self.steps = steps
                self.T_niters = T_niters
                self.alpha = alpha 
                self.bound = bound
                self.device = device
                self.mu = mu
                self.gamma = gamma
                self.target_layer = target_layer
                self.bound = bound

        def attack(self, model, ori_img, label):


                x_adv = torch.zeros_like(ori_img)

                for t in range(self.T_niters):
                        x_adv.requires_grad_(True)
                        model_children = torchvision.models._utils.IntermediateLayerGetter(model, {self.target_layer: 'feature'})
                        feature_ori = model_children(ori_img)['feature']
                        feature_adv = model_children(x_adv)['feature']
                        loss = torch.sum(torch.norm(feature_adv - feature_ori, p=2, dim = [1,2,3])) / x_adv.shape[0]

                        loss.backward()

                        grad = x_adv.grad

                        x_adv = x_adv + self.alpha * torch.sign(grad)

                        x_adv = torch.where(x_adv > ori_img + self.epsilon, ori_img + self.epsilon, x_adv)
                        x_adv = torch.where(x_adv < ori_img - self.epsilon, ori_img - self.epsilon, x_adv)

                        x_adv[:,0,:,:]= torch.clamp(x_adv[:,0,:,:], self.bound[0], self.bound[1])
                        x_adv[:,1,:,:]= torch.clamp(x_adv[:,1,:,:], self.bound[2], self.bound[3])
                        x_adv[:,2,:,:]= torch.clamp(x_adv[:,2,:,:], self.bound[4], self.bound[5])
                        x_adv = x_adv.detach()

                return x_adv




if __name__ == '__main__':

        steps = args.steps
        T_niters = args.T_niters
        alpha = args.alpha / (255 * 0.225)
        epsilon = args.epsilon /(255 * 0.225)
        mu = args.mu
        gamma = args.gamma
        target_layer = args.target_layer
        model_name = args.model_name
        batch_size = args.batch_size
        save_dir = args.save_dir


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

        model.eval()
        model.to(device)


        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        norm = T.Normalize(tuple(mean), tuple(std))
        resize = T.Resize(tuple((input_size[1:])))

        bound = compute_bound(mean, std)

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

        ori_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = 0, pin_memory = False)

        attack_type = NRDM(epsilon, alpha, steps, T_niters, mu, gamma, target_layer, bound, device)

        label_ls = []
        correct = 0
        for ind, (ori_img , label) in enumerate(ori_loader):
                label_ls.append(label)
                ori_img = ori_img.to(device)
                label = label.to(device)

                img_adv = attack_type.attack(model, ori_img, label)

                print('successful')


                predict = model(ori_img)
                predict_adv = model(img_adv)
                predicted = torch.max(predict.data, 1)[1]
                predicted_adv = torch.max(predict_adv.data, 1)[1]
                correct += (predicted_adv == label).sum().cpu().numpy()

                print('label:', label)
                print('predict:', predicted)
                print('predict_adv:', predicted_adv)
                print(correct)
                img_adv = inv_img(img_adv, mean, std)

                np.save(save_dir + '/batch_{}.npy'.format(ind), torch.round(img_adv.data*255).cpu().numpy().astype(np.uint8()))

                del ori_img, img_adv
                print('batch_{}.npy saved'.format(ind))

        label_ls = torch.cat(label_ls)
        np.save(save_dir + '/labels.npy', label_ls.numpy())
        print('images saved')


