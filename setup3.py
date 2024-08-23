import torch
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.projects import densepose

# Initialize Detectron2 with DensePose
cfg = get_cfg()
densepose.add_densepose_config(cfg)
cfg.merge_from_file(model_zoo.get_config_file("densepose_rcnn_R_50_FPN_s1x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("densepose_rcnn_R_50_FPN_s1x.yaml")

# Create predictor
predictor = DefaultPredictor(cfg)

# Load image and run DensePose
image = cv2.imread("your_image.png")
outputs = predictor(image)

# The rest of your processing goes here...

# Display the image
cv2.imshow("DensePose Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
