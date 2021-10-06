from teeth_final import  *
from resnet_gnn import  *
from torch_geometric.data import Data
from PIL import ImageColor
import torch

teeth_colors = {
'1': '#ffe2e2',
'2': '#ffc6c6',
'3': '#ffaaaa',
'4': '#ff8d8d',
'5': '#ff7171',
'6': '#ff5555',
'7': '#ff3838',
'8': '#ff0000',
'9': '#0000ff',
'10': '#3838ff',
'11': '#5555ff',
'12': '#7171ff',
'13': '#8d8dff',
'14': '#aaaaff',
'15': '#c6c6ff',
'16': '#e2e2ff',
}

def process_images(img_list):
    images = []
    for img in img_list:
        input_image = Image.fromarray(img[:, :, ::-1])
        input_image = input_image.resize((256, 256))
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image)
        images.append(input_tensor)

    return torch.stack(images)

def get_data_object(data_dict, graph_transforms):
    label = []
    images_list = []
    pos_list = []
    for k in data_dict.keys():
        lbl = int(k)-1
        if lbl > 13 or lbl < 0:
            continue
        label.append(lbl)
        pos_list.append([data_dict[k]['c_x'], data_dict[k]['c_y']])
        images_list.append(data_dict[k]['image'])
    images = process_images(images_list)
    labels = torch.tensor(label)
    pos = torch.tensor(pos_list)
    data = Data(x=images, y=labels, pos=pos)
    t_data = graph_transforms(data)
    return t_data

def draw_result(image, data_dict, pred):
    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    text_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    for i, k in enumerate(data_dict.keys()):
        color = ImageColor.getcolor(teeth_colors[str(int(k)+1)], "RGB")

        x_coords = np.array(data_dict[k]['x'])
        y_coords = np.array(data_dict[k]['y'])
        pts = np.vstack([x_coords, y_coords]).astype(np.int32).T
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], color)


        x = int(data_dict[k]['c_x'])
        y = int(data_dict[k]['c_y'])
        lbl = int(pred[i])+1
        cv2.putText(text_mask, "GT:"+str(k), (x-20, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
        cv2.putText(text_mask, "Pred:"+str(lbl), (x-20,y+10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    cv2.addWeighted(mask, 0.5, image, 0.9, 0, image)
    cv2.addWeighted(text_mask, 1, image, 0.7, 0, image)
    # cv2.imshow('', image)
    # cv2.waitKey(0)
    return image



if __name__ == '__main__':
    num_classes = 14
    input_dir = "/home/ahmed/workspace/data/upper_occlusal_188"
    input_dir = "/home/ahmed/workspace/data/occlusal_upper/pre_processed"
    # input_dir = "/home/ahmed/workspace/data/gnn_test_data"
    device = 'cpu' #torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    graph_transforms = transforms.Compose([T.KNNGraph(k=3), T.Cartesian()])
    checkpoint_path = "/home/ahmed/workspace/teeth_classification_2d_gnn/output/model_24.pth"
    model = Resnet_GNN(num_classes=num_classes, device=device).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device=device))
    model.load_state_dict(checkpoint['model_state_dict'])

    json_file_list = getJsonFile(input_dir)
    for json_fname in tqdm(json_file_list[::-1]):
        try:
            data_dict, image = extract_teeth(json_fname)
            data = get_data_object(data_dict, graph_transforms)
            data = data.to(device=device)
            out = model(data.x, data.edge_index, data.edge_attr)
            pred = out.argmax(dim=1)
            image = draw_result(image, data_dict, pred)
            cv2.imshow('', image)
            cv2.waitKey(0)

        except:
            print("skiped : " + json_fname)
            continue







