from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps
import cv2 as cv
import numpy as np
import os
import albumentations as alm

type = 'type_3'
dataset = 'train'
parent_dir = '/home/vkorneychuk/projects/lum/photo/03.11/white/blades/{}/{}/'.format(dataset, type)
mask_dir = '/home/vkorneychuk/projects/lum/photo/03.11/white/segmentation_blades/{}/{}/'.format(dataset, type)
files = os.listdir(parent_dir)
max_idx = len(files)-1
curr = 0
thresh_min = 40
orig = cv.imread(parent_dir + files[curr], cv.IMREAD_COLOR)
cv.namedWindow('processed', cv.WINDOW_NORMAL)
cv.namedWindow('transformed', cv.WINDOW_NORMAL)

while True:
    # Display auto-contrast version of corresponding target (per-pixel categories)
    img = PIL.ImageOps.autocontrast(PIL.Image.fromarray(np.uint8(orig)))
    img = np.asarray(img)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(img, thresh_min, 255, cv.THRESH_BINARY)
    cv.imshow('processed', mask)

    pushed_key = cv.waitKey(1)
    if pushed_key == 13:
        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        max_cnt = max(contours, key=cv.contourArea)
        max_mask = np.zeros(img.shape, np.uint8)
        cv.fillPoly(max_mask, pts=[max_cnt], color=(255, 255, 255))
        # res = cv.bitwise_and(mask, max_mask)
        cv.imshow('mask', max_mask)

        cv.imwrite(mask_dir + files[curr], max_mask)
        print('Saved to: {}'.format(mask_dir + files[curr]))
        curr = curr + 1 if curr < max_idx else 0
        orig = cv.imread(parent_dir + files[curr], cv.IMREAD_COLOR)
        print(files[curr])
    elif pushed_key == 83:
        curr = curr + 1 if curr < max_idx else 0
        orig = cv.imread(parent_dir + files[curr], cv.IMREAD_COLOR)
        cv.imshow('orig', orig)
        print(files[curr])
    elif pushed_key == 81:
        curr = curr - 1 if curr > 0 else max_idx
        orig = cv.imread(parent_dir + files[curr], cv.IMREAD_COLOR)
        cv.imshow('orig', orig)
        print(files[curr])
    elif pushed_key == 43:
        thresh_min += 5
        print(thresh_min)
    elif pushed_key == 45:
        thresh_min -= 5
        print(thresh_min)
    # elif pushed_key == 99:
    #     cv.Canny()
    elif pushed_key == 27:
        break
    elif pushed_key == 116:
        transform = alm.Compose([
            alm.ShiftScaleRotate(rotate_limit=360),
            alm.ElasticTransform()
        ])
        transformed = transform(image=img, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask']
        cv.imshow('transformed', np.hstack([transformed_image, transformed_mask]))
        # transformed_mask = transformed['mask']
    elif pushed_key != -1:
        print(pushed_key)