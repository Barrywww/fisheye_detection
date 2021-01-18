import pickle
import os
import numpy as np
from skimage import io, draw
from math import floor


DATA_DIR = "data/wireframe/pointlines/"

def visualize(filename):
    with open(DATA_DIR + filename + '.pkl', 'rb') as file:
        data = pickle.load(file)
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
            print(y, x)

        io.imshow(im)
        io.show()

def info(filename):
    with open(DATA_DIR + 'pointlines/' + filename + '.pkl', 'rb') as file:
        data = pickle.load(file)
        print(data.keys())


filename = '00030043'
visualize(filename)
