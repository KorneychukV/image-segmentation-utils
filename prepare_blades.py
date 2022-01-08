import random

import numpy as np
import cv2 as cv
import os

type = 'type_4'
parent_dir = '/home/vkorneychuk/projects/lum/photo/03.11/white/{}_dataset/'.format(type)
train_dir = '/home/vkorneychuk/projects/lum/photo/03.11/white/blades/train/{}/'.format(type)
test_dir = '/home/vkorneychuk/projects/lum/photo/03.11/white/blades/test/{}/'.format(type)

files = os.listdir(parent_dir)
i = 0

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

test_num  = int(len(files) * 0.2)
test_idxs = random.sample(range(0, len(files)-1), test_num)

for i, file in enumerate(files):
    img = cv.imread(parent_dir+file, cv.IMREAD_GRAYSCALE)
    new_h, new_w = img.shape
    scale_coeff = 16
    scaled_image = cv.resize(img, (256,256), interpolation=cv.INTER_CUBIC)
    if file[9] == '2':
        scaled_image = np.rot90(scaled_image, 2)

    if i in test_idxs:
        cv.imwrite(test_dir+'{}_{:03d}.png'.format(type,i), scaled_image[:, :350])
        print(test_dir+'{}_{:03d}.png'.format(type,i))
    else:
        cv.imwrite(train_dir + '{}_{:03d}.png'.format(type,i), scaled_image[:, :350])
        print(train_dir + '{}_{:03d}.png'.format(type, i))
    print(file + ' compressed')
