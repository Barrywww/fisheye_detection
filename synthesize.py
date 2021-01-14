import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte, io, transform
import skimage
import random

OUT_RADIUS = 450
OUT_DIAMETER = OUT_RADIUS * 2
focal_lengths = [8, 9, 10]
per_mm = 3.7795275591
min_fl = min(focal_lengths) * per_mm


def crop_center(im: np.ndarray):
    w, h = im.shape[:2]
    res = np.abs((w - h) // 2)
    if w > h:
        im = im[res: -res, :]
    else:
        im = im[:, res:-res]
    
    return transform.resize(im, (OUT_DIAMETER, OUT_DIAMETER))

def crop_circle(im: np.ndarray):
    im = crop_center(im)
    alpha = np.zeros((im.shape[0], im.shape[1], 1)).astype(int)
    rr, cc = skimage.draw.disk((OUT_RADIUS, OUT_RADIUS), OUT_RADIUS)
    alpha[rr, cc] = 1
    out = np.dstack((im, alpha))
    return out

def new_focal_length():
    fl_mm = random.choice(focal_lengths)
    fl_px = fl_mm * per_mm
    return fl_mm, fl_px

def fisheye(xy):
    f = new_focal_length()[1]
    max_theta = np.arctan(OUT_DIAMETER / f)

    center = np.mean(xy, axis=0)
    x, y = (xy - center).T * 2

    r = np.sqrt(x**2 + y**2)
    theta = max_theta * r / OUT_RADIUS
    d = f * np.tan(theta)
    phi = np.arctan2(y, x)
    
    return np.column_stack((d * np.cos(phi), d * np.sin(phi))) + center


im = io.imread('/Users/tyawang/OneDrive/Research/fisheye_detection/playground/data/000000000139.jpg')

processed_im = im
out = transform.warp(processed_im, fisheye)

f, (ax0, ax1) = plt.subplots(1, 2, subplot_kw=dict(xticks=[], yticks=[]))
ax0.imshow(processed_im)
ax1.imshow(out)

plt.show()
