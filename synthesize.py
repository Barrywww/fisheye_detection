import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte, io, transform
import skimage
import random
import pickle
from wireframe import draw_points

DATA_DIR = "data/wireframe/pointlines/"
OUT_RADIUS = 500
OUT_DIAMETER = OUT_RADIUS * 2

focal_lengths = [50]
per_mm = 3.7795275591
min_fl = min(focal_lengths) * per_mm


def crop_rect(img: np.ndarray, points: np.ndarray, length=OUT_DIAMETER):
    w, h = img.shape[:2]
    c0 = np.array((w, h)) / 2
    points = points - c0

    res = np.abs((w - h) // 2)
    if w > h:
        img = img[res: -res, :]
    elif w < h:
        img = img[:, res: -res]
    
    k = length / img.shape[0]
    points = points * k + length / 2
    return transform.resize(img, (length, length)), points

def resize(img: np.ndarray, points: np.ndarray, min_length=OUT_DIAMETER):
    w, h = img.shape[:2]

    if min(w, h) < OUT_DIAMETER:
        c0 = np.array((w, h)) / 2
        points = points - c0
        if w < h:
            k = OUT_DIAMETER / w
            out = transform.resize(img, (OUT_DIAMETER, int(h * k)))
        else:
            k = OUT_DIAMETER / h
            out = transform.resize((int(k * w), OUT_DIAMETER))
        points = points * k + np.array(out.shape[:2]) // 2

    return out, points

def crop_circle(im: np.ndarray, points=None, radius=OUT_RADIUS, rect=True):
    if rect:
        im, points = crop_rect(im, points)

    center = (im.shape[0]//2, im.shape[1]//2)
    rr, cc = skimage.draw.disk(center, radius)

    out = np.zeros(im.shape)
    out[rr, cc] = im[rr, cc]
    return out, points

def new_focal_length():
    fl_mm = random.choice(focal_lengths)
    fl_px = fl_mm * per_mm
    return fl_mm, fl_px

def get_img_pointline(data):
    lines = np.array(data['lines'])
    points = np.array(data['points'])
    points = np.column_stack((points[:, 1], points[:, 0]))
    img = data['img']
    return img, points, lines

def warp_points(points: np.ndarray, center, scale, f):
    points -= center
    points *= scale

    u, v = points[:, 0], points[:, 1]
    phi = np.arctan2(v, u)
    d = np.sqrt(u**2 + v**2)
    theta = np.arctan2(d, f)
    r = 2 * theta * OUT_RADIUS / np.pi

    return np.column_stack((r * np.cos(phi), r * np.sin(phi))) + center

def fisheye_transform(data: np.ndarray, circle=True):
    img, points, lines = get_img_pointline(data)
    
    if circle:
        img, points = crop_circle(img, points)
    else:
        img, points = resize(img, points)
    
    f = new_focal_length()[1]

    uc, vc = np.array(img.shape[:2]) // 2
    u_max, v_max = np.array(img.shape[0]) - uc, np.array(img.shape[1]) - vc

    d = np.sqrt(u_max**2 + v_max**2)
    theta0 = np.arctan(d / f)
    phi0 = np.arctan(v_max / u_max)
    r = 2 * theta0 * OUT_RADIUS / np.pi
    scale = OUT_RADIUS / r

    warped_points = warp_points(points, (uc, vc), scale, f)
    out = transform.warp(img, _fisheye, map_args={'f': f, 'scale': scale})

    # TODO: solve scaling for non-circled pictures
    
    return out, warped_points

def _fisheye(xy, f, scale):
    center = np.mean(xy, axis=0)
    xy = (xy - center) / scale
    x, y = xy[:, 0], xy[:, 1]

    r = np.sqrt(x**2 + y**2)
    theta = np.pi / 2 * r / OUT_RADIUS
    d = f * np.tan(theta)
    phi = np.arctan2(y, x)
    
    return np.column_stack((d * np.cos(phi), d * np.sin(phi))) + center


if __name__ == '__main__':
    filename = '00030043'
    with open(DATA_DIR + filename + '.pkl', 'rb') as file:
        data = pickle.load(file)
        img, points, lines = get_img_pointline(data)

        out, warped_points = fisheye_transform(data)

        draw_points(img, points)
        draw_points(out, warped_points, size=4, rgb_scale=1)

        f, (ax0, ax1) = plt.subplots(1, 2, subplot_kw=dict(xticks=[], yticks=[]))
        ax0.imshow(img)
        ax1.imshow(out)

        plt.show()
