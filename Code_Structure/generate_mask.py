# -*- coding:utf-8 -*-
# coding:utf-8
import os
import cv2
import numpy as np
from interval import Interval

np.random.seed(2020)

def sort_axis(axis,stand):
    small_axis = []
    large_axis = []
    for a in axis:
        if a[1] < stand:
            small_axis.append(a)
        else:
            large_axis.append(a)
    
    if small_axis != []:
        for i in range(len(small_axis)):
            for j in range(0,len(small_axis)-i-1):
                if small_axis[j][0] > small_axis[j+1][0] :
                    small_axis[j], small_axis[j+1] = small_axis[j+1], small_axis[j]
    if large_axis != []:
        for i in range(len(large_axis)):
            for j in range(0,len(large_axis)-i-1):
                if large_axis[j][0] < large_axis[j+1][0] :
                    large_axis[j], large_axis[j+1] = large_axis[j+1], large_axis[j]
    newaxis =  small_axis + large_axis
    return newaxis

# compute known area
def compute_area(mask):
    area = 0
    height, width = mask.shape
    for i in range(height):
        for j in range(width):
            if mask[i, j] == 1:   # unknown mask area to generate
                area += 1
    return area / (height * width)



def random_inout(image_shape, m_scale, m_mv, m_ma, m_ml, m_mbw):
    """
    Generate a random free form mask for in/outpainting.
    """

    h,w = image_shape[0], image_shape[1]
    mask = np.ones((h,w))
    scale = m_scale
    outnum = 10
    innum = 6
    middle_h = h // 2
    # zoom_large = Interval(0,min(h,w))
    zoom_middle_x = Interval(w//4,w * 3 //4)
    zoom_middle_y = Interval(h//4,h * 3 //4)
    zoom_small_x = Interval(w//2 - w//16,w//2 + w//16)
    zoom_small_y = Interval(h//2 - h//16,h//2 + h//16)
    outaxis = []
    inaxis = []
    while True:
        for i in range(outnum):
            while 1:
                out_x = np.random.randint(w)
                if i < outnum // 2:
                    out_y = np.random.randint(h//6)
                else:
                    out_y = np.random.randint(h*4//6, high=h)
                # out_y = np.random.randint(h)
                if out_x not in zoom_middle_x and out_y not in zoom_middle_y:
                    outaxis.append([out_x,out_y])
                    break

        outaxis = sort_axis(outaxis,middle_h)
        # print("outaxis: ",outaxis)
        cv2.fillPoly(mask, np.array([outaxis], dtype = np.int32), 0)

        ## random polygon inpainting mask
        # for j in range(innum):
        #     in_x = np.random.randint(min(h,w)//2 - min(h,w)//6,min(h,w)//2 + min(h,w)//6)
        #     if j < innum // 2:
        #         in_y = np.random.randint(min(h,w)//2 - min(h,w)//6, high=min(h,w)//2)
        #     else:
        #         in_y = np.random.randint(min(h,w)//2,high=min(h,w)//2 + min(h,w)//6)
        #     # in_y = np.random.randint(min(h,w)//2 - min(h,w)//8,min(h,w)//2 + min(h,w)//8)
        #     inaxis.append([in_x,in_y])
        # inaxis = sort_axis(inaxis,middle_h)
        # # print("inaxis: ",inaxis)
        # cv2.fillPoly(mask, np.array([inaxis], dtype = np.int32), 1)

        # random freeform inpainting mask
        num_v = m_mv #tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)
        for i in range(num_v):
            start_x = np.random.randint(w//4,high=w*5//6)
            start_y = np.random.randint(h//4,high=h*5//6)
            angle = 0.01+np.random.randint(m_ma)
            # if i % 2 == 0:
            #     angle = 2 * 3.1415926 - angle
            angle = 2 * 3.1415926 - angle
            length = 20+np.random.randint(m_ml)    # 100 + for 512
            brush_w = 1+np.random.randint(m_mbw)   # 20 for 256/512
            end_x = (start_x + length * np.sin(angle)).astype(np.int32) 
            end_y = (start_y + length * np.cos(angle)).astype(np.int32)
            # end_x = end_x if end_x in zoom_middle_x else np.random.randint(w * 3//6,high=w* 5//6)
            # end_y = end_y if end_y in zoom_middle_y else np.random.randint(h * 3//6,high=h* 5//6)
            # end_x = np.random.randint(w * 5//6,high=w)
            # end_y = np.random.randint(h * 5//6,high=h)
            cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)
            # print("end_x: ",end_x, "end_y: ",end_y)
            cal_scale = compute_area(mask)
            # print(" cal_scale: ",cal_scale)
            if scale-0.1 <= cal_scale  and  cal_scale < scale+0.1:
                # print(" finish cal_scale: ",cal_scale)  # cal_scale:  0.495849609375
                break        
        
        cal_scale = compute_area(mask)
        print(" cal_scale: ",cal_scale)
        if scale-0.1 <= cal_scale  and  cal_scale < scale+0.1:
            print(" finish cal_scale: ",cal_scale)  # cal_scale:  0.495849609375
            break
        # print(countnum, "mask generate ",cal_scale)

    # print("mask finish")
    # time.sleep(1)
    return mask.reshape((1,)+mask.shape).astype(np.float32)



# def random_inout(image_shape, m_scale, m_mv, m_ma, m_ml, m_mbw):
#     """
#     Generate a random free form mask for in/outpainting.
#     """

#     h,w = image_shape[0], image_shape[1]
#     mask = np.ones((h,w))
#     scale = m_scale
#     outnum = 8
#     innum = 6
#     middle_h = h // 2
#     zoom_large = Interval(0,min(h,w))
#     zoom_middle = Interval(min(h,w)//4,min(h,w) * 3 //4)
#     zoom_small = Interval(min(h,w)//2 - min(h,w)//16,min(h,w)//2 + min(h,w)//16)
#     outaxis = []
#     inaxis = []
#     countnum = 0
#     while True:
#         countnum +=1 
#         for i in range(outnum):
#             while 1:
#                 out_x = np.random.randint(w)
#                 if i < outnum // 2:
#                     out_y = np.random.randint(h//6)
#                 else:
#                     out_y = np.random.randint(h*5//6, high=h)
#                 # out_y = np.random.randint(h)
#                 if out_x not in zoom_middle and out_y not in zoom_middle:
#                     outaxis.append([out_x,out_y])
#                     break

#         outaxis = sort_axis(outaxis,middle_h)
#         # print("outaxis: ",outaxis)
#         cv2.fillPoly(mask, np.array([outaxis], dtype = np.int32), 0)

#         ## random polygon inpainting mask
#         # for j in range(innum):
#         #     in_x = np.random.randint(min(h,w)//2 - min(h,w)//6,min(h,w)//2 + min(h,w)//6)
#         #     if j < innum // 2:
#         #         in_y = np.random.randint(min(h,w)//2 - min(h,w)//6, high=min(h,w)//2)
#         #     else:
#         #         in_y = np.random.randint(min(h,w)//2,high=min(h,w)//2 + min(h,w)//6)
#         #     # in_y = np.random.randint(min(h,w)//2 - min(h,w)//8,min(h,w)//2 + min(h,w)//8)
#         #     inaxis.append([in_x,in_y])
#         # inaxis = sort_axis(inaxis,middle_h)
#         # # print("inaxis: ",inaxis)
#         # cv2.fillPoly(mask, np.array([inaxis], dtype = np.int32), 1)

#         # random freeform inpainting mask
#         num_v = m_mv #tf.random_uniform([], minval=0, maxval=config.MAXVERTEX, dtype=tf.int32)
#         for i in range(num_v):
#             start_x = np.random.randint(min(h,w)//6,high=min(h,w)*5//6)
#             start_y = np.random.randint(min(h,w)//6,high=min(h,w)*5//6)
#             angle = 0.01+np.random.randint(m_ma)
#             # if i % 2 == 0:
#             #     angle = 2 * 3.1415926 - angle
#             angle = 2 * 3.1415926 - angle
#             length = 30+np.random.randint(m_ml)
#             brush_w = 15+np.random.randint(m_mbw)
#             end_x = (start_x + length * np.sin(angle)).astype(np.int32) 
#             end_y = (start_y + length * np.cos(angle)).astype(np.int32)
#             end_x = end_x if end_x in zoom_middle else np.random.randint(min(h,w)//4,high=min(h,w) * 3//4)
#             end_y = end_y if end_y in zoom_middle else np.random.randint(min(h,w)//4,high=min(h,w) * 3//4)
#             cv2.line(mask, (start_y, start_x), (end_y, end_x), 1.0, brush_w)

#             cal_scale = compute_area(mask)
#             # print(" cal_scale: ",cal_scale)
#             if scale-0.1 <= cal_scale  and  cal_scale < scale+0.1:
#                 # print(" finish cal_scale: ",cal_scale)  # cal_scale:  0.495849609375
#                 break        
        
#         cal_scale = compute_area(mask)
#         print(" cal_scale: ",cal_scale)
#         if scale-0.1 <= cal_scale  and  cal_scale < scale+0.1:
#             print(" finish cal_scale: ",cal_scale)  # cal_scale:  0.495849609375
#             break
#         # print(countnum, "mask generate ",cal_scale)

#     # print("mask finish")
#     # time.sleep(1)
#     return mask.reshape((1,)+mask.shape).astype(np.float32)



if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from PIL import Image

    parser = argparse.ArgumentParser()
    # parser.add_argument('--input_dir', default='./input_dir', type=Path)
    parser.add_argument('--output_dir', '-o', default=None, type=Path)
    parser.add_argument('--H', default=256, type=int)              # image H size, 256 by default.
    parser.add_argument('--W', default=256, type=int)              # image w size, 256 by default.  256/512
    parser.add_argument('--m_scale', default=0.1, type=float)      # Range (0, 1), 0.5 by default.   0.1/0.3
    parser.add_argument('--m_mv', default=50, type=int)            # Range (0, 50], 50 by defalut. 
    parser.add_argument('--m_ma', default=4, type=int)             # angle, 4 by defalut. 
    parser.add_argument('--m_ml', default=20, type=int)            # length, 10 by default.
    parser.add_argument('--m_mbw', default=10, type=int)           # width, 20 by default.
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = './output_dir_256' + '_' + str(args.m_scale)
        os.makedirs(args.output_dir, exist_ok=True)

    # files = os.listdir(args.input_dir)
    # print("Total number of images: ",len(files))

    image_shape = (args.H, args.W, 3)
    imges_num = 31000                        # CelebA 31000  Paris 16000
    
    for i  in range(imges_num):
        # print("img: ",img)
        # image_in = os.path.join(args.input_dir, img)
        # image = np.array(Image.open(image_in))
        
        # smoothed = tsmooth(image, args.lambda_, args.sigma, args.sharpness, args.max_iter)
        # image_save = os.path.join(args.output_dir, img)
        # Image.fromarray((smoothed * 256).astype('uint8')).save(str(image_save))

        mask = random_inout(image_shape, args.m_scale, args.m_mv, args.m_ma, args.m_ml, args.m_mbw)
        img_name = str(i+1) + '.jpg'
        image_save = os.path.join(args.output_dir, img_name)
        # print(i, " mask shape: ",mask.shape)
        mask = mask.transpose((1,2,0))*255
        print(i, " mask shape: ",mask.shape)
        # print("mask: ",mask)
        cv2.imwrite(image_save,mask)


    
    print('*'*6," Finish ",'*'*6)