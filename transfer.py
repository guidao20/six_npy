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



if torch.cuda.is_available():
        device = torch.device('cuda')
else:
        device = torch.device('cpu')

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

class AdvImagenet(Dataset):
    def __init__(self, imagepath, labelpath, transform=None):
        super(AdvImagenet, self).__init__()
        self.imagepath = imagepath
        self.labelpath = labelpath
        self.transform = transform
        self._load_data()
    def _load_data(self):
        self.images = np.load(self.imagepath)
        self.labels = np.load(self.labelpath)
    def __getitem__(self, item):
        image = Image.fromarray(self.images[item].transpose(1,2,0))
        label = self.labels[item]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.images)



model_name = 'densenet'
                       
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


batch_size = 4
model.eval()
model.to(device)

if model_name in ['squeezenet', 'resnet', 'densenet', 'senet' ,'vgg']:
	    input_size = [3, 224, 224]
else:
	    input_size = [3, 299, 299]

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

trans_adv = T.Compose([
            resize,
            T.ToTensor(),
            norm
    ])



dataset = SelectedImagenet(imagenet_val_dir='data/imagenet/ILSVRC2012_img_val',
                           selected_images_csv='data/imagenet/selected_imagenet.csv',
                           transform=trans
                           )
ori_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = False)

correct = 0


advpath = 'NAA_SI_CONV.npy'
imagepath = os.path.join('result', advpath)
labelpath = os.path.join('result', 'labels.npy')

advdataset = AdvImagenet(imagepath = imagepath, labelpath = labelpath, transform = trans_adv)
adv_loader = torch.utils.data.DataLoader(advdataset, batch_size=batch_size, shuffle=False, num_workers = 8, pin_memory = False)

for ind, (adv_img , label) in enumerate(adv_loader):
        adv_img = adv_img.to(device)
        label = label.to(device)
        predict = model(adv_img)
        predicted = torch.max(predict.data, 1)[1]
        correct += (predicted == label).sum()                
        print(predicted, label, correct)  


# for i in range(num - 1):
#         path_npy = os.path.join(path, 'batch_' + str(i) + '.npy')
#         images = np.load(path_npy) / 255.0
#         images.dtype = 'float32'
#         labels = labels_np[i*4:(i+1)*4]
#         images = torch.from_numpy(images).to(device)
#         labels = torch.from_numpy(labels).to(device)
#         output  = model(images)

#         predicted = torch.max(output.data, 1)[1]
#         correct += (predicted == labels).sum()                

#for ind, (ori_img , label) in enumerate(ori_loader):
#        ori_img = ori_img.to(device)
#        label = label.to(device)
#        predict = model(ori_img)
#        predicted = torch.max(predict.data, 1)[1]
#        correct += (predicted == label).sum()                
#        print(correct)          
