import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte, io, transform
import skimage
import random

OUT_RADIUS = 450
OUT_DIAMETER = OUT_RADIUS * 2
focal_lengths = [40]
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
    rr, cc = skimage.draw.disk((OUT_RADIUS, OUT_RADIUS), OUT_RADIUS)

    out = np.zeros(im.shape)
    out[cc, rr] = im[cc, rr]
    return out

def new_focal_length():
    fl_mm = random.choice(focal_lengths)
    fl_px = fl_mm * per_mm
    return fl_mm, fl_px

def fisheye_transform(img: np.ndarray):
    f = new_focal_length()[1]
    uc, vc = np.array(img.shape[:2]) // 2

    xmax = np.array(img.shape[0])

    d = np.sqrt(uc**2 + vc**2)
    theta0 = np.arctan(d / f)
    phi0 = np.arctan(vc / uc)
    r = 2 * theta0 * OUT_RADIUS / np.pi
    x_scaled= r * np.cos(phi0) + uc
    return transform.warp(img, _fisheye, map_args={'f': f, 'x_scaled': x_scaled, 'x_max': xmax})

def _fisheye(xy, f, x_scaled, x_max):
    # max_theta = np.arctan(OUT_DIAMETER / f)
    max_theta = np.pi / 2

    center = np.mean(xy, axis=0)
    x, y = (xy - center).T * x_max // 2 // x_scaled * 1.5

    r = np.sqrt(x**2 + y**2)
    theta = max_theta * r / OUT_RADIUS
    d = f * np.tan(theta)
    phi = np.arctan2(y, x)
    
    return np.column_stack((d * np.cos(phi), d * np.sin(phi))) + center


im = io.imread('data/000000000139.jpg')

processed_im = crop_circle(im)
# processed_im = im
out = fisheye_transform(processed_im)

f, (ax0, ax1) = plt.subplots(1, 2, subplot_kw=dict(xticks=[], yticks=[]))
ax0.imshow(processed_im)
ax1.imshow(out)

plt.show()
