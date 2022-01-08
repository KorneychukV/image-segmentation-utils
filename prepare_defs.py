import random

import numpy as np
import cv2 as cv
import os
import albumentations as alm

# type = 'defs'
# parent_dir = '/home/vkorneychuk/projects/lum/photo/03.11/white/defs/'
# train_dir = '/home/vkorneychuk/projects/lum/photo/03.11/white/def_dataset/train/images/'
# test_dir = '/home/vkorneychuk/projects/lum/photo/03.11/white/def_dataset/test/images/'

# files = os.listdir(parent_dir)
# i = 0

# test_num = int(len(files) * 0.2)
# test_idxs = random.sample(range(0, len(files)-1), test_num)

# for i, file in enumerate(files):
#     img = cv.imread(parent_dir+file, cv.IMREAD_GRAYSCALE)
#     new_h, new_w = img.shape
#     scale_coeff = 16
#     scaled_image = cv.resize(img, (256,256), interpolation=cv.INTER_CUBIC)
#
#     if i in test_idxs:
#         cv.imwrite(test_dir+'def_{:03d}.png'.format(i), scaled_image[:, :350])
#         print(test_dir+'def_{:03d}.png'.format(i))
#     else:
#         cv.imwrite(train_dir + 'def_{:03d}.png'.format(i), scaled_image[:, :350])
#         print(train_dir + 'def_{:03d}.png'.format(i))
#     print(file + ' compressed')


parent_dir_img = '/home/vkorneychuk/projects/lum/photo/03.11/white/def_dataset/train/images1/'
parent_dir_mask = '/home/vkorneychuk/projects/lum/photo/03.11/white/def_dataset/train/masks1/'
train_dir_img = '/home/vkorneychuk/projects/lum/photo/03.11/white/def_dataset/train/images/'
train_dir_mask = '/home/vkorneychuk/projects/lum/photo/03.11/white/def_dataset/train/masks/'

imgs = os.listdir(parent_dir_img)
masks = os.listdir(parent_dir_mask)

i = 0
for img_name, mask_name in zip(imgs, masks):
    img = cv.imread(parent_dir_img+img_name, cv.IMREAD_GRAYSCALE)
    mask = cv.imread(parent_dir_mask+mask_name, cv.IMREAD_GRAYSCALE)

    for _ in range(30):
        transform = alm.Compose([
            alm.ShiftScaleRotate(rotate_limit=360),
            alm.ElasticTransform()
        ])
        transformed = transform(image=img, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']

        cv.imwrite(train_dir_img + 'def_{:03d}.png'.format(i), transformed_image)
        cv.imwrite(train_dir_mask + 'def_{:03d}.png'.format(i), transformed_mask)
        print('Saved def_{:03d}.png'.format(i))
        print('Saved def_{:03d}.png'.format(i))
        i += 1
