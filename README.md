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
