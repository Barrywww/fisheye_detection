# Preparation

Put image into a `data/` folder in the root directory and change `filename` in the script you want to run.

# Synthesizing

Should we:
- Resize the input image?
- Crop the input image to circle? If so, how large?

Things to decide on:
- parameter: OUT_RADIUS
- focal lengths

## Running `synthesize.py`

The synthesizing pipeline:
1. crop input image to rectangle of side length 2 * OUT_RADIUS
2. crop rectangle to circle of radius OUT_RADIUS
3. transform to circular fisheye image with a random focal length in the focal_lengths array
4. crop the rest mapping in the corner we don't want
