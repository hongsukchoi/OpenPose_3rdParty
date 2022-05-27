import os
import glob
import json
import re
import sys
sys.path.append('.')
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from tqdm import tqdm

from lib.network.rtpose_vgg import get_model
from lib.network import im_transform
from evaluate.coco_eval import get_outputs, handle_paf_and_heat
from lib.utils.common import Human, BodyPart, CocoPart, CocoColors, CocoPairsRender, draw_humans, parse_human_to_array
from lib.utils.paf_to_pose import paf_to_pose_cpp
from lib.config import cfg, update_config

def vis_keypoints(img, kps, alpha=1, kps_vis=None):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for i in range(len(kps)):
        p = kps[i][0].astype(np.int32), kps[i][1].astype(np.int32)
        cv2.circle(kp_mask, p, radius=3, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)
        if kps_vis is not None:
            cv2.putText(kp_mask, str(kps_vis[i, 0]), p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        else:
            cv2.putText(kp_mask, str(i), p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Blend the keypoints.
    return cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', help='experiment configure file name',
                    default='./experiments/vgg19_368x368_sgd.yaml', type=str)
parser.add_argument('--weight', type=str,
                    default='pose_model.pth')
parser.add_argument('opts',
                    help="Modify config options using the command-line",
                    default=None,
                    nargs=argparse.REMAINDER)
args = parser.parse_args()

# update config file
update_config(cfg, args)



model = get_model('vgg19')     
model.load_state_dict(torch.load(args.weight))
model = torch.nn.DataParallel(model).cuda()
model.float()
model.eval()

#test_image = './readme/ski.jpg'
input_dir = '/home/hongsukchoi/projects/Data/MultiPersonTestSet'#TS1/img_000000.jpg'
seq_list = glob.glob(input_dir + '/*')

for seq in seq_list:
    seq_result = {}
    seq_name = seq.split('/')[-1]
    img_path_list = glob.glob(seq + '/*.jpg')

    for img_path in tqdm(img_path_list):
        img_name = img_path.split('/')[-1]
        img = cv2.imread(img_path)  # B,G,R order
        image_h, image_w = img.shape[:2]

        with torch.no_grad():
            paf, heatmap, im_scale = get_outputs(img, model, 'rtpose')

        humans = paf_to_pose_cpp(heatmap, paf, cfg)
        img_result = parse_human_to_array(humans, image_h, image_w)
        seq_result[img_name] = img_result.tolist()

    save_path = f'openpose_result_{seq_name}.json'
    with open(save_path, 'w') as f:
        json.dump(seq_result, f)
    print("Saved at ", save_path)
