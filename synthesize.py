import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
import skimage
import random
import pickle
import os
from wireframe import draw_points, draw_point_lines

DATA_DIR = "data/wireframe/"
OUT_RADIUS = 500
OUT_DIAMETER = OUT_RADIUS * 2

focal_lengths = [50]
per_mm = 3.7795275591
min_fl = min(focal_lengths) * per_mm


def crop_rect(img: np.ndarray, points=None, length=OUT_DIAMETER):
    w, h = img.shape[:2]

    res = np.abs((w - h) // 2)
    if w > h:
        img = img[res: -res, :]
    elif w < h:
        img = img[:, res: -res]

    if points is not None:
        c0 = np.array((w, h)) / 2
        points = points - c0
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


def get_heatmap(img, points, lines, f, scale):
    points += np.array(img.shape[:2]) // 2 - 1

    orig_hm = np.zeros(img.shape[:2])

    for i in range(lines.shape[0]):
        start, end = points[lines[i][0]].astype(int),\
                            points[lines[i][1]].astype(int)
        # print(start, end)
        distance = np.linalg.norm(start - end)
        rr, cc = skimage.draw.line(start[0], start[1], end[0], end[1])
        line = np.column_stack((rr, cc))
        line = line[(line[:, 0] >= 0) & (line[:, 1] < img.shape[1])]
        rr, cc = line[:, 0], line[:, 1]
        orig_hm[rr, cc] = distance

    fisheye_hm = transform.warp(orig_hm, _fisheye,
                                map_args={'f': f})
    fisheye_hm = crop_corners(fisheye_hm, scale)

    return fisheye_hm


def crop_corners(img: np.ndarray, scale):
    length0 = img.shape[0]
    length_prime = int(scale * length0)
    out = transform.resize(img, (length_prime, length_prime))
    res = (length_prime - length0) // 2
    out = out[res:-res, res:-res]
    return out


def warp_points(points: np.ndarray, center, f, scale):
    points -= center

    u, v = points[:, 0], points[:, 1]
    phi = np.arctan2(v, u)
    d = np.sqrt(u**2 + v**2)
    theta = np.arctan2(d, f)
    r = 2 * theta * OUT_RADIUS / np.pi

    return np.column_stack((r * np.cos(phi) * scale,
                            r * np.sin(phi) * scale)) + center


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
    r = 2 * theta0 * OUT_RADIUS / np.pi

    scale = OUT_RADIUS / r
    out = transform.warp(img, _fisheye, map_args={'f': f})
    out = crop_corners(out, scale)
    warped_points = warp_points(points, (uc, vc), f, scale)

    fisheye_hm = get_heatmap(img, points, lines, f, scale)

    return out, warped_points, f, fisheye_hm


def _fisheye(xy, f):
    center = np.mean(xy, axis=0)
    xy = (xy - center)
    x, y = xy[:, 0], xy[:, 1]

    r = np.sqrt(x**2 + y**2)
    theta = np.pi / 2 * r / OUT_RADIUS
    d = f * np.tan(theta)
    phi = np.arctan2(y, x)

    return np.column_stack((d * np.cos(phi), d * np.sin(phi))) + center


def visualize_example():
    filename = '00030043'
    with open(DATA_DIR + 'pointlines/' + filename + '.pkl', 'rb') as file:
        data = pickle.load(file)
        img, points, lines = get_img_pointline(data)

        out, warped_points, f, fisheye_hm = fisheye_transform(data)

        draw_points(img, points)
        draw_point_lines(img, points, lines)
        draw_points(out, warped_points, size=4, rgb_scale=1)
        draw_point_lines(out, warped_points, lines, rgb_scale=1)

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3,
                                            subplot_kw=dict(
                                                xticks=[],
                                                yticks=[]))
        ax0.imshow(img)
        ax1.imshow(out)
        ax2.imshow(fisheye_hm)

        plt.show()


def synthesize():
    filenames = os.listdir(DATA_DIR + 'pointlines/')
    for filename in filenames:
        with open(os.path.join(DATA_DIR, 'pointlines/',
                               filename), 'rb') as file:
            data = pickle.load(file)
            img, points, lines = get_img_pointline(data)
            fisheye, warped_points, f, fisheye_hm = fisheye_transform(data)

            fisheye_data = {
                'filename': filename,
                'img': img,
                'fisheyeImg': fisheye,
                'focalLength': f,
                'points': points,
                'lines': lines,
                'warpedPoints': warped_points,
                'fisheyeHeatmap': fisheye_hm
            }

            with open(os.path.join(DATA_DIR,
                                   'fisheye_pointlines/',
                                   filename), 'wb') as out_file:
                pickle.dump(fisheye_data, out_file,
                            protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    visualize_example()
