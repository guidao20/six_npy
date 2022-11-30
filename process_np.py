import os
import numpy as np


path = os.path.join(os.getcwd(), 'test')
path_list = os.listdir(path)
num = len(path_list)


image_list = []

for i in range(num - 1):
    path_npy = os.path.join(path, 'batch_' + str(i) + '.npy')
    batch_image = np.load(path_npy)
    for j in range(batch_image.shape[0]):   
        image_list.append(batch_image[j])

image = np.array(image_list)
np.save(os.path.join('result','NAA_SI_CONV.npy'),image)

path = os.path.join('result','NAA_SI_CONV.npy')

input_ = np.load(path)
print(input_.shape)
