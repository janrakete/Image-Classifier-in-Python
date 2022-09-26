#imports
from torchvision import models
from torchvision import transforms
from torchvision import datasets

import torch
from torch import nn
from torch import optim
import torch.nn.functional as torch_functional

from collections import OrderedDict 
import numpy as np
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import json
import random
import os

import argparse

#argument parser
argument_parser = argparse.ArgumentParser(description = 'Prediction argument parser')

argument_parser.add_argument('--input',           action = 'store', type = str,  default = './flowers/test/13/image_05745.jpg')
argument_parser.add_argument('--category_names',  action = 'store', type = str,  default = 'cat_to_name.json')
argument_parser.add_argument('--top_k',           action = 'store', type = int,  default = 5)
argument_parser.add_argument('--gpu',             action = 'store', type = str,  default = 'gpu')
argument_parser.add_argument('--checkpoint_path', action = 'store', type = str,  default = './checkpoint.pth')

args = argument_parser.parse_args()

input_file      = args.input
category_names  = args.category_names
top_k           = args.top_k
gpu             = args.gpu
checkpoint_path = args.checkpoint_path

device = torch.device("cuda:0" if torch.cuda.is_available() and gpu == 'gpu' else "cpu") # use cuda if available and wanted

#defining all global variables
global_categories_file = category_names

global_transforms_random_rotation = 20
global_transforms_resize = 256
global_transforms_center_crop = 224
global_transforms_random_resized_crop = 224

global_normalize_mean = [0.485, 0.456, 0.406]
global_normalize_standard = [0.229, 0.224, 0.225]

#function to crop and resize image
def image_process(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image = Image.open(image)    

    data_trans_single = transforms.Compose([transforms.Resize(global_transforms_resize),
                                           transforms.CenterCrop(global_transforms_center_crop),
                                           transforms.ToTensor(),
                                           transforms.Normalize(global_normalize_mean,
                                                                global_normalize_standard)])
    
    image = data_trans_single(image)

    return image


#main
def main():
    
    #label mapping from file
    with open(global_categories_file, 'r', encoding = 'utf-8') as file:
        category_to_name = json.load(file)

    print ("Loaded categories:")
    print(category_to_name)

    #loading the checkpoint
    checkpoint = torch.load(checkpoint_path)

    if  checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained = True) #load pretrained neural netwwork
    else:
        model = models.vgg16(pretrained = True) #load pretrained neural netwwork

    model.to(device)                
        
    for param in model.parameters(): 
        param.requires_grad = False  

    model.class_to_idx =    checkpoint['class_to_idx']
    model.classifier =      checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])    
    model.idx_to_class =    checkpoint['idx_to_class'] #from mentor
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), checkpoint['learning_rate'])        
    
    #predict
    image_path = input_file

    image = image_process(image_path)
    print ("#####")
    print("Test flower: " + category_to_name[os.path.basename(os.path.dirname(image_path))])    
    print ("#####")
    print ("Predictions:")
    
    model.eval()
    
    image = image.unsqueeze_(0)
    image = image.float()
    
    with torch.no_grad():
        output = model.forward(image.to(device))
        
    probs = torch_functional.softmax(output.data, dim = 1)

    largest = probs.topk(top_k) #from mentor
    
    probs = largest[0].cpu().numpy()[0] #from mentor
    y = np.array(probs)
    
    idx = largest[1].cpu().numpy()[0] #from mentor
    classes = [model.idx_to_class[x] for x in idx]     #from mentor
    x = []
    for index in np.array(classes):
        x.append(category_to_name[str(index)])
    
    index = 0
    while index < len(y):
        print("-> Flower: {} , Probability: {}".format(x[index], y[index]))
        index += 1
    
if __name__== "__main__":
    main()    
    
    
    
