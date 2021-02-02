from bs4 import BeautifulSoup
import os
from tqdm import tqdm


ANNOTATION_DIR = "data/VOC_360/Annotations/"
OUT_DIR = "data/VOC360-yolo/labels/"
names = ['person','bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']


for filename in tqdm(os.listdir(ANNOTATION_DIR)):
    with open(ANNOTATION_DIR + filename, "r") as xml:
        soup = BeautifulSoup(xml, "html.parser")
        outname = soup.filename.string.split('.')[0] + ".txt"
        with open(OUT_DIR + outname, 'w') as outfile:
            objects = soup.find_all("object")
            imwidth = int(soup.size.width.string)
            imheight = int(soup.size.height.string)
            for object in objects:
                bndbox = object.find("bndbox", recursive=False)
                if bndbox is not None:
                    name = str(names.index(object.find('name').string))
                    xmin = int(bndbox.find("xmin").string)
                    xmax = min(imwidth, int(bndbox.xmax.string))
                    ymin = int(bndbox.ymin.string)
                    ymax = min(imheight, int(bndbox.ymax.string))
                    xcenter = str((xmin + xmax) / 2 / imwidth)
                    ycenter = str((ymin + ymax) / 2 / imheight)
                    width = str((xmax - xmin) / imwidth)
                    height = str((ymax - ymin) / imheight)
                    line = ' '.join((name, xcenter, ycenter, width, height)) + '\n'
                    outfile.write(line)
