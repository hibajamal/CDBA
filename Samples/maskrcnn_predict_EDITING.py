# USAGE
# python maskrcnn_predict.py --weights mask_rcnn_coco.h5 --labels coco_labels.txt --image images/30th_birthday.jpg

# import the necessary packages
import numpy as np
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time
from PIL import Image

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True,
	help="path to Mask R-CNN model weights pre-trained on COCO")
ap.add_argument("-l", "--labels", required=True,
	help="path to class labels file")
ap.add_argument("-i", "--image", required=True,
	help="path to input image to apply Mask R-CNN to")
args = vars(ap.parse_args())

# load the class label names from disk, one label per line
CLASS_NAMES = open(args["labels"]).read().strip().split("\n")

start = time.time()

# generate random (but visually distinct) colors for each class label
# (thanks to Matterport Mask R-CNN for the method!)

class SimpleConfig(Config):
	# give the configuration a recognizable name
	NAME = "coco_inference"

	# set the number of GPUs to use along with the number of images
	# per GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# number of classes (we would normally add +1 for the background
	# but the background class is *already* included in the class
	# names)
	NUM_CLASSES = len(CLASS_NAMES)

# initialize the inference configuration
config = SimpleConfig()

# initialize the Mask R-CNN model for inference and then load the
# weights
print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config,
	model_dir=os.getcwd())
model.load_weights(args["weights"], by_name=True)

# load the input image, convert it from BGR to RGB channel
# ordering, and resize the image
image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = imutils.resize(image, width=512)

# perform a forward pass of the network to obtain the results
print("[INFO] making predictions with Mask R-CNN...")
r = model.detect([image], verbose=1)[0]

# create empty variable to store mask of human detected
mask = None

# loop over of the detected object's bounding boxes and masks
for i in range(0, r["rois"].shape[0]):
	# extract the class ID and mask for the current detection, then
	# grab the color to visualize the mask (in BGR format)
	classID = r["class_ids"][i]
	mask = r["masks"][:, :, i]

	if CLASS_NAMES[classID] == "person":
		color = (0, 0, 0)
		# first person detected is our target
		break

# adding alpha channel
image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
# converting numpy image to pillow image
pil_image = Image.fromarray(image)
# creating new pil image for png
qhuman_mask = Image.new(pil_image.mode, pil_image.size)
# loading both images
new_pixels = qhuman_mask.load()
original_pixels = pil_image.load()

# create the PNG image pixel by pixel
for i in range(qhuman_mask.size[0]):
	for j in range(qhuman_mask.size[1]):
		if mask[j][i]:
			new_pixels[i, j] = original_pixels[i, j]
		else:
			new_pixels[i, j] = (0, 0, 0, 0)

qhuman_mask.save("out.png")
qhuman_mask.close()

print()
print(time.time()-start, "seconds taken")

cv2.waitKey()
