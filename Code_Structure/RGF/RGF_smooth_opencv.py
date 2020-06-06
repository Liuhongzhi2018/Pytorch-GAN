# -*- coding:utf-8 -*-
# coding:utf-8
import os
import cv2
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from PIL import Image

    import cv2 as cv2 
    from matplotlib import pyplot as plt
    from cv2.ximgproc import rollingGuidanceFilter

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='./input_dir', type=Path)
    parser.add_argument('--output_dir', '-o', default=None, type=Path)
    parser.add_argument('--d', default=-1, type=int)              # -1 by default.
    parser.add_argument('--sigma_c', default=50, type=float)      # 25 by default.
    parser.add_argument('--sigma_s', default=10, type=float)      # 3 by defalut. 
    parser.add_argument('--iteration', default=3, type=int)       # Number of itearations, 4 by default.
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = './output_dir'
        os.makedirs(args.output_dir, exist_ok=True)

    din, sc, ss, ite = args.d, args.sigma_c, args.sigma_s, args.iteration

    files = os.listdir(args.input_dir)
    print("Total number of images: ",len(files))
    
    for img in files:
        print("img: ",img)
        image_in = os.path.join(args.input_dir, img)
        image = cv2.imread(image_in)
        # print("image in: ",image)
        smoothed = rollingGuidanceFilter(image, d=din, sigmaColor=sc, sigmaSpace=ss, numOfIter=ite)
        # print("smoothed: ",smoothed)
        save_path = os.path.join(args.output_dir, img)
        # Image.fromarray((smoothed * 256).astype('uint8')).save(str(image_save))
        cv2.imwrite(save_path, smoothed)   # jpg default 95
    
    print('*'*6," Finish ",'*'*6)


# Run code
# python RGF_smooth.py --d -1 --sigma_c 25 --sigma_s 3 --iteration 5

# def rollingGuidanceFilter(src, dst=None, d=None, sigmaColor=None, sigmaSpace=None, numOfIter=None, borderType=None):
# src: 输入图像；
# d: 滤波半径，如果是负数的话，则从sigmaSpace计算出来； diameter of each pixel neighborhood
# sigmaColor: 色彩权重，类似于双边滤波核值域因子； color mix, larger area with bigger color 
# sigmaSpace: 空域权重； farther pixels will influence each other with larger value
# numOfIter: 迭代次数； iterations of joint edge-preserving filter
# borderType: 边界填充方式；
