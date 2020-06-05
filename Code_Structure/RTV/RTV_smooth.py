import os
import cv2
import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve


def tsmooth(I, lambda_=0.01, sigma=3.0, sharpness=0.02, max_iter=4):
    I = im2double(I)
    x = I
    sigma_iter = sigma
    lambda_ = lambda_ / 2
    dec = 2

    for i in range(max_iter):
        wx, wy = compute_texture_weights(x, sigma_iter, sharpness)
        x = solve_linear_equation(I, wx, wy, lambda_)
        sigma_iter /= dec
        if sigma_iter < 0.5:
            sigma_iter = 0.5

    return x

def im2double(image):
    return image / 255

def compute_texture_weights(fin, sigma, sharpness):
    fx = np.diff(fin, 1, 1)
    fx = np.pad(fx, [(0,0),(0,1),(0,0)], 'constant')
    fy = np.diff(fin, 1, 0)
    fy = np.pad(fy, [(0,1),(0,0),(0,0)], 'constant')

    vareps_s = sharpness
    vareps = 0.001

    wto = np.maximum(((fx**2 + fy**2) ** 0.5).mean(axis=-1), vareps_s) ** (-1)
    fbin = lpfilter(fin, sigma)
    gfx = np.diff(fbin, 1, 1)
    gfx = np.pad(gfx, [(0,0),(0,1),(0,0)], 'constant')
    gfy = np.diff(fbin, 1, 0)
    gfy = np.pad(gfy, [(0,1),(0,0),(0,0)], 'constant')
    wtbx = np.maximum(np.abs(gfx).mean(axis=-1), vareps) ** (-1)
    wtby = np.maximum(np.abs(gfy).mean(axis=-1), vareps) ** (-1)
    retx = wtbx * wto
    rety = wtby * wto

    retx[:, -1] = 0
    rety[-1, :] = 0

    return retx, rety

# ref at https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def conv2_sep(im, sigma):
    ksize = round(5 * sigma) | 1
    g = matlab_style_gauss2D((1,ksize), sigma)
    ret = cv2.filter2D( im, -1,   g, borderType=cv2.BORDER_CONSTANT)
    ret = cv2.filter2D(ret, -1, g.T, borderType=cv2.BORDER_CONSTANT)
    return ret

def lpfilter(image, sigma):
    filtered = np.array([conv2_sep(array, sigma) for array in image.transpose((2,0,1))])
    filtered = filtered.transpose((1,2,0))
    return filtered

def solve_linear_equation(in_, wx, wy, lambda_):
    r, c, ch = in_.shape
    k = r * c
    dx = -lambda_ * wx.flatten('F')
    dy = -lambda_ * wy.flatten('F')
    B = np.stack((dx, dy))
    d = (-r, -1)
    A = spdiags(B, d, k, k)
    e = dx
    w = np.pad(dx, [(r,0)], 'constant')[:-r]
    s = dy
    n = np.pad(dy, [(1,0)], 'constant')[:-1]
    D = 1 - (e+w+s+n)
    A = A + A.T + spdiags(D, 0, k, k)

    out = np.zeros((r, c, ch))
    for i in range(ch):
        tin = in_[..., i].flatten('F')
        tout = spsolve(A, tin)
        out[..., i] += tout.reshape(c, r).T

    return out

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='./input_dir', type=Path)
    parser.add_argument('--output_dir', '-o', default=None, type=Path)
    parser.add_argument('--lambda_', default=0.01, type=float)
    parser.add_argument('--sigma', default=3.0, type=float)
    parser.add_argument('--sharpness', default=0.02, type=float)
    parser.add_argument('--max_iter', default=4, type=int)
    args = parser.parse_args()

    # if args.output_img is None:
    #     l = args.lambda_
    #     s = args.sigma
    #     # Path.parent -> return logical father root
    #     # Path.stem -> return item without suffix
    #     args.output_img = args.input_img.parent / \
    #         (args.input_img.stem + f'_sigma_{int(s):d}_lambda_{l:.3f}' + args.input_img.suffix)

    if args.output_dir is None:
        args.output_dir = './output_dir'
        os.makedirs(args.output_dir, exist_ok=True)

    files = os.listdir(args.input_dir)
    print("Total number of images: ",len(files))
    
    for img in files:
        print("img: ",img)
        image_in = os.path.join(args.input_dir, img)
        image = np.array(Image.open(image_in))
        smoothed = tsmooth(image, args.lambda_, args.sigma, args.sharpness, args.max_iter)
        image_save = os.path.join(args.output_dir, img)
        Image.fromarray((smoothed * 256).astype('uint8')).save(str(image_save))
    
    print('*'*6," Finish ",'*'*6)


# Run code
# python RTV_smooth.py --lambda_ 0.01 --sigma 3 --sharpness 0.02 --max_iter 4