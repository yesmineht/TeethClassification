import os
import json
import numpy as np
import cv2
import glob
from tqdm import tqdm


def getJsonFile(path, extension='.json'):
    pattern = path + '/*' + extension
    list = glob.glob(pattern)
    return list



def check_orientation(data_dict):
    incisors = []
    molars = []
    molars_1_5 = []
    molars_10_14 = []

    for poly in data['polygon']:
        x = np.array(poly['coords']['all_x'])
        y = np.array(poly['coords']['all_y'])
        pts = np.vstack([x, y]).astype(int).T
        try:
            lbl = int(poly['label ID'])
        except:
            continue
        if lbl in [1, 2, 3, 4, 5]:
            molars_1_5.append(np.mean(pts, axis=0))
        if lbl in [10, 11, 12, 13, 14]:
            molars_10_14.append(np.mean(pts, axis=0))
        if lbl in [6, 7, 8, 9]:
            incisors.append(np.mean(pts, axis=0))

    try:
        mean_incisors = np.mean(np.vstack(incisors), axis=0)
        mean_molars = np.mean(np.vstack([molars_1_5, molars_10_14]), axis=0)
        mean_molars_1_5 = np.mean(np.vstack(molars_1_5), axis=0)
        mean_molars_10_14 = np.mean(np.vstack(molars_10_14), axis=0)
        orientation = 'up'
        if mean_incisors[1] > mean_molars[1]:
            orientation = 'down'
        if mean_molars_1_5[0] < mean_molars_10_14[0]:
            orientation += '_lr'
        else:
            orientation += '_rl'
    except:
        orientation = 'up_lr'

    return orientation

def get_affine_matrix(ori, H, W):
    ori = ori.split('_')
    matrix = np.eye(2,3)
    if ori[0] == 'up' and ori[1] == 'rl':
        matrix[0, 0] = -1
        matrix[0, 2] = W - 1
    if ori[0] == 'down' and ori[1] == 'lr':
        matrix[1, 1] = -1
        matrix[1, 2] = H - 1
    return matrix

if __name__ == "__main__":
    input_dir = "/home/ahmed/workspace/data/upper_occlusal_188"
    input_dir = "/home/ahmed/workspace/data/occlusal_upper"
    # input_dir = "/home/ahmed/workspace/data/occlusal_data"
    export_dir = os.path.join(input_dir, 'pre_processed')
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)


    json_file_list = getJsonFile(input_dir)

    for json_fname in json_file_list:
        print(json_fname)
        # try:
        fname = os.path.basename(json_fname).split('.')[0]
        # Opening JSON file
        f = open(os.path.join(input_dir, json_fname))
        data = json.load(f)
        new_data = data.copy()
        # if len(data['polygon']) > 14 or len(data['polygon']) < 2:
        #     raise Exception
        ori = check_orientation(data)
        # read image correspondante lel json
        image = cv2.imread(os.path.join(input_dir, data['imName']))
        H, W = image.shape[:2]
        tr_matrix = get_affine_matrix(ori, H, W)
        image = cv2.warpAffine(image, tr_matrix, (W, H))
        # cv2.imshow('src', image)
        # cv2.imshow('dst', dst)
        # cv2.waitKey(0)
        dict = {}
        # Iterating through the json
        pts_list = []
        for new_poly, poly in zip(new_data['polygon'], data['polygon']):
            x = np.array(poly['coords']['all_x'])
            y = np.array(poly['coords']['all_y'])
            pts = np.vstack([x, y]).astype(int).T
            pts = (tr_matrix @ np.vstack([pts.T, np.array([1] * pts.shape[0]).reshape((1, -1))])).T
            pts = pts.astype(int)
            new_poly['coords']['all_x'] = pts[:, 0].tolist()
            new_poly['coords']['all_y'] = pts[:, 1].tolist()
            try:
                lbl = int(poly['label ID'])
                if lbl > 14 :
                    lbl = lbl - 14
                new_poly['label ID'] = str(lbl)
            except:
                pass
            # draw labels
            #
            # try:
            #     lbl = int(poly['label ID'])
            #     cx, cy = pts.mean(axis=0).astype(int)
            #     cv2.putText(image, str(lbl), (cx, cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1)
            # except:
            #     pass


        # export
        with open(os.path.join(export_dir, os.path.basename(json_fname)), "w") as outfile:
            json.dump(new_data, outfile)
        cv2.imwrite(os.path.join(export_dir, data['imName']), image)
        # except:
        #     print("skiped : " + fname)
        #     continue
