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
import copy

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

# generate random (but visually distinct) colors for each class label
# (thanks to Matterport Mask R-CNN for the method!)


'''
	THESE NEXT 4 LINES ARE IN CHARGE OF MASK COLORS
'''


'''the next line for hsv has to be changed for colors to become binary for our TWO classifiers
hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]
COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
the next two lines should be omitted 
random.seed(42)
random.shuffle(COLORS)'''

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

masked_image = copy.deepcopy(image) # added
cv2.namedWindow("HumanDetected") # added
cv2.namedWindow("HumanMask") # added
mask = None # create empty variable to store mask of human detected

# loop over of the detected object's bounding boxes and masks
for i in range(0, r["rois"].shape[0]):
	# extract the class ID and mask for the current detection, then
	# grab the color to visualize the mask (in BGR format)
	classID = r["class_ids"][i]
	mask = r["masks"][:, :, i]
	print(type(mask), mask.shape, mask.dtype)

	if CLASS_NAMES[classID] == "person":
		color = (0, 0, 0)

		# visualize the pixel-wise mask of the object
		# image = visualize.apply_mask(image, mask, color, alpha=1)
		masked_image = visualize.apply_mask(masked_image, mask, color, alpha=1)
		print(masked_image.shape, type(masked_image))

		(startY, startX, endY, endX) = r["rois"][i] # added
		masked_image = masked_image[startY:endY, startX:endX]
		break # added till here

# convert the image back to BGR so we can use OpenCV's drawing
# functions
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
masked_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY) # added
masked_image = cv2.bitwise_not(masked_image)

# loop over the predicted scores and class labels
'''
for i in range(0, len(r["scores"])):
	# extract the bounding box information, class ID, label, predicted
	# probability, and visualization color
	(startY, startX, endY, endX) = r["rois"][i]
	classID = r["class_ids"][i]
	label = CLASS_NAMES[classID]
	score = r["scores"][i]
	color = [int(c) for c in np.array(COLORS[classID]) * 255]

	# draw the bounding box, class label, and score of the object
	cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
	text = "{}: {:.3f}".format(label, score)
	y = startY - 10 if startY - 10 > 10 else startY + 10
	cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
		0.6, color, 2)
'''

# show the output image
# cv2.imshow("Output", image)
cv2.imshow("Input", image) # added
cv2.imshow("HumanDetected", masked_image) #added
cv2.waitKey()
