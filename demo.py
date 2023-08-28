import copy
import glob
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src import model
from src import util
from src.body import Body
from src.hand import Hand


def y_flip(bbox, height):
    # xyxy
    x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
    tmp_y1 = max(height - y2 - 1, 0)
    y2 = max(height - y1 - 1, 0)
    y1 = tmp_y1

    return [int(x1), int(y1), int(x2), int(y2)]

body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

folder = '/home/hongsuk.c/Projects/HandNeRF_annotation/data/handnerf_training_1/cam_*'# '/labdata/hongsuk/HandForce/Data/10042022_isaac/realsense/'  # './images'
files = sorted(glob.glob(folder + '/*_0010.jpg'))
print(len(files))
for file in sorted(files):
    test_image = file
    oriImg = cv2.imread(test_image)  # B,G,R order
    # candidate, subset = body_estimation(oriImg)
    # canvas = copy.deepcopy(oriImg)
    # canvas = util.draw_bodypose(canvas, candidate, subset)
    # detect hand
    # hands_list = util.handDetect(candidate, subset, oriImg)

    # load hand detection from external source
    ann_file_path = file.replace('jpg', 'json')
    with open(ann_file_path, 'r') as f:
        ann = json.load(f)
    hands_list= []
    for shape in ann["shapes"]:
        if "hand" in shape["label"] and "rectangle" == shape["shape_type"]:
            hb = np.array(shape["points"]).reshape(-1) # xy xy 
            is_left  = False
            # flip
            cam_idx = int(file.split('/')[-2].split('_')[-1]) 
            print(cam_idx)
            if cam_idx in [2, 3, 4, 5]:
                hb = y_flip(hb, oriImg.shape[0])
                is_left = True
                oriImg = oriImg[::-1]
                
            center = [(hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2]
            width = max(hb[2] - hb[0], hb[3] - hb[1]) * 2.0

            hands_list.append([int(center[0] - width/2), int(center[1] - width/2), int(width), is_left])
            # [int(x), int(y), int(width), is_left
            
    if len(hands_list) == 0:
        import pdb; pdb.set_trace()


    all_hand_peaks = []
    canvas = copy.deepcopy(oriImg)
    for x, y, w, is_left in hands_list:
        cv2.rectangle(canvas, (x, y), (x+w, y+w), (0, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.putText(canvas, 'left' if is_left else 'right', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # if is_left:
            # plt.imshow(oriImg[y:y+w, x:x+w, :][:, :, [2, 1, 0]])
            # plt.show()
        peaks = hand_estimation(oriImg[y:y+w, x:x+w, :])
        
        peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
        peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        # else:
        #     peaks = hand_estimation(cv2.flip(oriImg[y:y+w, x:x+w, :], 1))
        #     peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], w-peaks[:, 0]-1+x)
        #     peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)
        #     print(peaks)
        all_hand_peaks.append(peaks)

    canvas = util.draw_handpose(canvas, all_hand_peaks)

    plt.imshow(canvas[:, :, [2, 1, 0]])
    plt.axis('off')
    plt.show()
