# Preparation

Put image into a `data/` folder in the root directory and change `filename` in the script you want to run.

# Synthesizing

Should we:
- Resize the input image?
- Crop the input image to circle? If so, how large?
- Resize the output image smaller?

Things to decide on:
- parameter: OUT_RADIUS
- focal lengths

## Running `synthesize.py`

The synthesizing pipeline:
1. crop input image to rectangle of side length OUT_DIAMETER
2. crop rectangle to circle of radius OUT_RADIUS as input image to fisheye transformation
3. transform to circular fisheye image with a random focal length in the focal_lengths array, at the same time resize the fisheye part of the image
4. resize the entire image to OUT_DIAMETER * OUT_DIAMETER
5. crop the rest mapping in the corner we don't want

In `synthesize.py`: run `visualize_example()` to see what is happening to each picture.

To generate fisheye images, put the wireframe data set in folder `data/` with the following folder structure:

```
root/
|--data/
|---|---wireframe/
|---|----|----pointlines/
|---|----|-----|-----[all wireframe .pkl]
|---|----|----fisheye_pointlines/

```

And the output information will be generated into `fisheye_pointlines/`.

In every .pkl file generated, the following information is contained:
1. filename: the filename of the original .pkl
2. img: the original input image
3. fisheyeImg: the generated fisheye image in the form of numpy array
4. focalLength: the corresponding focal length associated to the synthesizing so that the picture can be dewarped
5. points: points associated with the original picture
6. lines: lines needed by the network
7. warpedPoints: points after warpping
8. fisheyeHeatmap: ground truth for fisheye heatmap
