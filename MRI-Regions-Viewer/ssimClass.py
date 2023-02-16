import pathlib
import numpy as np
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import xmltodict
import torch
from torchvision import datasets, transforms
from pytorch_grad_cam import GradCAM
from CNN_T import CNN_T

DATA_PATH = pathlib.Path(__file__).parent.joinpath("data").resolve()


#functions
def colorMapping(x, name_map="magma"):
    cm = plt.get_cmap(name_map)

    x_map1 = cm(x[ 0, :, :,].detach().numpy())
    
    x_map2 = torch.Tensor(x_map1[ :, :, :3]).to(torch.float).transpose(-2, -1).transpose(0, 1)

    return x_map2

def get_cam(model, x, target_layers):
    cam = GradCAM(model=model, target_layers=target_layers)
    
    grayscale_cam = cam(input_tensor=x)
    return grayscale_cam

def getHeatmap(selected_stage):
    transform_normal = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    colorMapping
                                ])
    #load image to imageFolder
    if selected_stage == "Non Demented":
        dataset = datasets.ImageFolder("./data/ImageFolders/NonDemented/", transform=transform_normal)
    elif selected_stage == "Very Mild Demented":
        dataset = datasets.ImageFolder("./data/ImageFolders/VeryMildDemented/", transform=transform_normal)
    elif selected_stage == "Mild Demented":
        dataset = datasets.ImageFolder("./data/ImageFolders/MildDemented/", transform=transform_normal)
    else:
        dataset = datasets.ImageFolder("./data/ImageFolders/ModerateDemented/", transform=transform_normal)
    
    image = dataset[0][0]
    inp_data = np.ndarray([100, 3, 224, 224])
    inp_data[0] = image.detach().numpy()
    inp_data = torch.tensor(inp_data)

    #load the model
    model = CNN_T(
        lr = 0.0017377061778563376,
        batch_size= 100,
        head_dim= 16,
        mhsa_n_dim= 21,
        multilayer_perceptron_dim= 29,
        dropout_p= 0.0024779216175651675
    )
    model.create_model()
    model.load_state_dict(torch.load("./data/CNN_T-Modell.pth"))

    #get the heatmap
    target_layers = [model.convNet.conv1, model.convNet.conv2, model.convNet.conv3, model.convNet.conv4]
    heatmaps = get_cam(model, inp_data.to(torch.float), target_layers=target_layers[:])
    image_heatmap = heatmaps[0]

    cm = plt.get_cmap("jet")
    image_heatmap = cm(image_heatmap)

    return image_heatmap


def get_contours(image, index=None):
    #img_gray = cv2.resize(image, (128, 128))
    #img_gray = np.array(image, np.uint8)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 90, 255, cv2.THRESH_BINARY)
    #plt.imshow(thresh)
    img_thresh, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours != 0 and contours != ():
        
        if index == None:
            size = 0
            index = 0
            for count, i in enumerate(contours):
                if len(i) > size:
                    size = len(i)
                    index = count

        cnt = contours#[contours[index]]
        white = np.ndarray([image.shape[0], image.shape[1], image.shape[2]])
        image_contours = cv2.drawContours(image, cnt, -1, (0,255,0), 3)
        white_contours = cv2.drawContours(white, cnt, -1, (0,255,0), 3)
        #plt.imshow(image_contours)
        return white_contours, image_contours
    else:
        return None, None

def centerImage(img, t_top, t_down, t_left, t_right, extra_cut=None):
    top = 0
    down = img.shape[0]
    left = 0
    right = img.shape[1]
    for i in range(img.shape[0]):
        this_layer = img[i, :, :]
        if np.any(this_layer) == True and np.max(this_layer) > t_top:
            top = i
            break

    for i in reversed(range(img.shape[0])):
        this_layer = img[i, :, :]
        if np.any(this_layer) == True and np.max(this_layer) > t_down:
            down = i
            break
        
    for i in range(img.shape[1]):
        this_layer = img[:, i, :]
        if np.any(this_layer) == True and np.max(this_layer) > t_left:
            left = i
            break

    for i in reversed(range(img.shape[1])):
        this_layer = img[:, i, :]
        if np.any(this_layer) == True and np.max(this_layer) > t_right:
            right = i
            break
    
    if type(extra_cut).__module__ != np.__name__:
        return img[top:down, left:right]
    else:
        return img[top:down, left:right], extra_cut[top:down, left:right], [top, down, left, right]

def reverseCenterImage(img, values_list, original_shape):
    top = values_list[0]
    down = original_shape[0] - values_list[1]
    left = values_list[2]
    right = original_shape[1] - values_list[3]

    for i in range(top):
        img = img.copy()
        img = np.insert(img, 0, values=0, axis=0)
    
    for i in range(down):
        img = img.copy()
        img = np.insert(img, img.shape[0], values=0, axis=0)
    
    for i in range(left):
        img = img.copy()
        img = np.insert(img, 0, values=0, axis=1)
    
    for i in range(right):
        img = img.copy()
        img = np.insert(img, img.shape[1], values=0, axis=1)

    return img

def slice_img(selected_stage, selected_atlas, want_heatmap=False):
    heatmap = None
    heatmap_out = None

    #load image to compare to mrt
    if selected_stage == "Non Demented":
        image = cv2.imread(filename=str(DATA_PATH.joinpath("images/non.jpg")))
    elif selected_stage == "Very Mild Demented":
        image = cv2.imread(filename=str(DATA_PATH.joinpath("images/verymild.jpg")))
    elif selected_stage == "Mild Demented":
        image = cv2.imread(filename=str(DATA_PATH.joinpath("images/mild.jpg")))
    else:
        image = cv2.imread(filename=str(DATA_PATH.joinpath("images/moderate.jpg")))

    #get image contours
    img_contours, img_all_contours = get_contours(image)
    #center image contours
    img_contours, img_all_contours, size_con = centerImage(img_contours, 254, 254, 254, 254, img_all_contours)
    #img_all_contours = centerImage(img_all_contours, 254, 254, 2)

    #load mrt
    if selected_atlas == "mrt_regions":
        regions = nib.load(DATA_PATH.joinpath("1103_3_glm-test.nii"))
        regions_data = regions.get_fdata()
    else:
        regions = nib.load(DATA_PATH.joinpath("aal.nii.gz"))
        regions_data = regions.get_fdata()
        regions_data = np.rot90(regions_data, k=1, axes=(1,2))

    #print(regions_data.max())
    #print(regions_data.min())

    on_contours = []
    just_contours = []
    slice_values_list = []
    scores = []

    for i in range(regions_data.shape[1]):

        #get slice
        slice_triplet = regions_data[:, i, :]

        #make gray-image
        slice = cv2.merge([slice_triplet, slice_triplet, slice_triplet]).astype(np.uint8)

        #get contour with and without image
        white_contours, image_contours = get_contours(slice)
        
        if type(white_contours).__module__ == np.__name__:

            #get contours into right shape
            white_contours = white_contours.transpose(1, 0, 2)
            image_contours = image_contours.transpose(1, 0, 2)

            #save both

            #center white_contours
            white_contours, image_contours, slice_values = centerImage(white_contours, 254, 254, 254, 254, image_contours)
            slice_values_list.append(slice_values)
            if white_contours.shape[0] >= 7 & white_contours.shape[1]>=7:

                #resize image to white contours
                this_img = cv2.resize(img_contours, [white_contours.shape[1], white_contours.shape[0]])
                #calculate ssim for both
                score = ssim(white_contours[:, :, 1].astype("uint8"), this_img[:, :, 1].astype("uint8"))

                just_contours.append(white_contours)
                on_contours.append(image_contours)
                scores.append(score)
            else:
                scores.append(0)
                just_contours.append(np.zeros(slice.transpose(1, 0, 2).shape))
                on_contours.append(np.zeros(slice.transpose(1, 0, 2).shape))
        else:
            just_contours.append(np.zeros(slice.transpose(1, 0, 2).shape))
            on_contours.append(np.zeros(slice.transpose(1, 0, 2).shape))
            slice_values_list.append([])
            scores.append(0)

    max_score = max(scores)
    index = scores.index(max_score)

    img_contours = cv2.resize(img_contours, [just_contours[index].shape[1], just_contours[index].shape[0]])
    img_all_contours = cv2.resize(img_all_contours, [on_contours[index].shape[1], on_contours[index].shape[0]])
    #img_all_contours = cv2.resize(img_all_contours, [regions_data.shape[0], regions_data.shape[2]])

    #colored_img = reverseCenterImage(on_contours[index], slice_values_list[index], regions_data[:, index, :].T.shape)[:, :, 0]
    img_all_contours = reverseCenterImage(img_all_contours, slice_values_list[index], regions_data[:, index, :].T.shape)

    if want_heatmap:
        heatmap = getHeatmap(selected_stage=selected_stage)
        heatmap = cv2.resize(heatmap, [image.shape[0], image.shape[1]])
        heatmap = heatmap[size_con[0]:size_con[1], size_con[2]:size_con[3]]
        heatmap = cv2.resize(heatmap, [on_contours[index].shape[1], on_contours[index].shape[0]])
        heatmap_out = reverseCenterImage(heatmap, slice_values_list[index], regions_data[:, index, :].T.shape)


    #create labels
    if selected_atlas == "mrt_regions":
        with open(DATA_PATH.joinpath('1103_3_glm_LabelMap.xml'), 'r', encoding='utf-8') as file:
            my_xml = file.read()
        my_dict = xmltodict.parse(my_xml)
        new_dict = {"0": ""}
        for i in range(len(my_dict["LabelList"]["Label"])):
            number = my_dict["LabelList"]["Label"][i]["Number"]
            name = my_dict["LabelList"]["Label"][i]["Name"]
            new_dict[str(number)] = name

        colored_img = regions_data[:, index, :].T
        
        names = [[""]*colored_img.shape[1]]*colored_img.shape[0]
        names = np.array(names, dtype=np.object)
        for i in range(colored_img.shape[0]):
            for j in range(colored_img.shape[1]):
                #names = np.put(names, [i][j], new_dict[str(int(colored_img[i, j]))])
                names[i][j] = new_dict[str(int(colored_img[i, j]))]
    else:
        with open(DATA_PATH.joinpath('aal.nii.txt')) as f:
            lines = f.readlines()
        value_list = [""]
        for count, line in enumerate(lines):
            value_list.append(line.split()[1])

        colored_img = regions_data[:, index, :].T

        names = [[""]*colored_img.shape[1]]*colored_img.shape[0]
        names = np.array(names, dtype=np.object)
        for i in range(colored_img.shape[0]):
            for j in range(colored_img.shape[1]):
                names[i][j] = value_list[int(colored_img[i, j])]

    """
    colored_img = regions_data[:, index, :].T
    slice_x = on_contours[index].shape[0]
    slice_y = on_contours[index].shape[1]
    colored_img = colored_img[int((slice_y/2)):int(colored_img.shape[0]-(slice_y/2)), int((slice_x/2)):int(colored_img.shape[1]-(slice_x/2))]
    colored_img = cv2.resize(colored_img, [regions_data.shape[0], regions_data.shape[2]])
    """
    #print(img_all_contours.shape)
    #print(colored_img.shape)

    return img_all_contours, colored_img, index, names, heatmap_out