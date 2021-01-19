import pickle
import numpy as np
from skimage import io, draw
import matplotlib.pyplot as plt
# from synthesize import DATA_DIR, crop_rect, resize

def draw_points(img, points, size=2, rgb_scale=255):
    for point in points:
        if size <= point[0] < img.shape[0] - size and size <= point[1] < img.shape[1] - size:
            img[draw.disk(point, size)] = (0, 1*rgb_scale, 0)

def draw_point_lines(img, points, lines):
    for line in lines:
        point1, point2 = line
        x1, y1 = int(points[point1][0]) - 1, int(points[point1][1]) - 1
        x2, y2 = int(points[point2][0]) - 1, int(points[point2][1]) - 1
        img[draw.line(x1, y1, x2, y2)] = (255, 0, 0)


def visualize(data):
    lines = data['lines']
    points = data['points']

    im = data['img']

    for idx, (i, j) in enumerate(lines, start=0):
        y1, x1 = int(points[i][0]) - 1, int(points[i][1]) - 1
        y2, x2 = int(points[j][0]) - 1, int(points[j][1]) - 1
        im[draw.line(x1, y1, x2, y2)] = (255, 0, 0)
    
    for idx, (y, x) in enumerate(data['junction']):
        x, y = int(x) - 1, int(y) - 1
        im[draw.disk((x, y), 2)] = (0, 255, 0)

    io.imshow(im)
    io.show()

def info(data):
    data['lines'] = np.array(data['lines'])
    print((data['lines']))

if __name__ == '__main__':
    filename = '00030043'
    with open("data/wireframe/pointlines/" + filename + '.pkl', 'rb') as file:
        data = pickle.load(file)
        info(data)
    #     points = np.array(data['points'])
    #     points = np.column_stack((points[:, 1], points[:, 0]))
    #     im = data['img']

        # out, warped_points = resize(im, points)
        # draw_points(out, warped_points, size=4, rgb_scale=1)
        # draw_points(im, points)
        
        # f, (ax0, ax1) = plt.subplots(1, 2, subplot_kw=dict(xticks=[], yticks=[]))
        # ax0.imshow(im)
        # ax1.imshow(out)

        # plt.show()
