import os
import json
import numpy as np
import cv2
import glob
from tqdm import tqdm

def get_teeth_image(image, pts):
    # mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.int32)
    # cv2.fillConvexPoly(mask, pts, 1)
    # mask = mask.astype(np.bool)
    # out = np.zeros_like(image)
    # out[mask] = image[mask]
    out = image

    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    c_x = (x1 + x2) // 2
    c_y = (y1 + y2) // 2
    w = x2 - x1
    h = y2 - y1
    crop_dim = max(w, h)
    x = int(c_x - crop_dim/2)
    y = int(c_y - crop_dim/2)
    tooth_image = out[y:y + crop_dim, x:x + crop_dim]

    return tooth_image

def getJsonFile(path, extension='.json'):
    pattern = path + '/*' + extension
    list = glob.glob(pattern)
    return list

def extract_teeth(json_fname):
    # Opening JSON file
    f = open(json_fname)
    data = json.load(f)
    # read image correspondante lel json
    dir_path = os.path.dirname(json_fname)
    image = cv2.imread(os.path.join(dir_path, data['imName']))
    dict = {}
    # Iterating through the json
    for poly in data['polygon']:
        x = np.array(poly['coords']['all_x'])
        y = np.array(poly['coords']['all_y'])
        pts = np.vstack([x, y]).astype(int).T
        tooth_image = get_teeth_image(image, pts)
        label = poly['label ID']
        dict[label] = {'x' : poly['coords']['all_x'],
                       'y' : poly['coords']['all_y'],
                       'c_x': int(np.mean(x)),
                       'c_y': int(np.mean(y)),
                       'image' : tooth_image}
    return dict, image


if __name__ == "__main__":
    input_dir = "/home/ahmed/workspace/data/upper_occlusal_188"
    input_dir = "/home/ahmed/workspace/data/occlusal_upper/pre_processed"
    export_dir = os.path.join(input_dir, 'raw')
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)


    json_file_list = getJsonFile(input_dir)

    for json_fname in tqdm(json_file_list):
        # try:
        fname = os.path.basename(json_fname).split('.')[0]
        # Opening JSON file
        f = open(os.path.join(input_dir, json_fname))
        data = json.load(f)
        if len(data['polygon']) > 14 or len(data['polygon']) < 2:
            continue

        output_dir = os.path.join(export_dir, fname)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # read image correspondante lel json
        image = cv2.imread(os.path.join(input_dir, data['imName']))
        dict = {}
        # Iterating through the json
        pts_list = []
        for poly in data['polygon']:
            x = np.array(poly['coords']['all_x'])
            y = np.array(poly['coords']['all_y'])
            pts = np.vstack([x, y]).astype(int).T
            pts_list.append(pts)
            tooth_image = get_teeth_image(image, pts)

            label = poly['label ID']
            try:
                lbl = int(label)
            except:
                continue
            if lbl not in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]:
                continue
            try:
                cv2.imwrite(os.path.join(output_dir, "{}.png".format(label)), tooth_image)
            except:
                continue
            dict[label] = {'c_x': int(np.mean(x)),
                           'c_y': int(np.mean(y))}
            # export dict to json file

        # compute normalized coordinates
        x1, y1 = np.vstack(pts_list).min(axis=0)
        x2, y2 = np.vstack(pts_list).max(axis=0)
        w = x2 - x1
        h = y2 - y1
        for k in dict.keys():
            dict[k]['nc_x'] = (dict[k]['c_x'] - x1) / w
            dict[k]['nc_y'] = (dict[k]['c_y'] - y1) / h

        # export
        with open(os.path.join(output_dir, 'coords.json'), "w") as outfile:
            json.dump(dict, outfile)
        # except:
        #     print("skiped : " + fname)
        #     continue
