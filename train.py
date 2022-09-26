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
import json

import argparse

#argument parser
argument_parser = argparse.ArgumentParser(description = 'Training argument parser')

argument_parser.add_argument('--data_dir',         action = 'store', type = str,   default = './flowers')
argument_parser.add_argument('--save_dir',         action = 'store', type = str,   default = './checkpoint.pth')
argument_parser.add_argument('--arch',             action = 'store', type = str,   default = 'vgg16')
argument_parser.add_argument('--learning_rate',    action = 'store', type = float, default = 0.001)
argument_parser.add_argument('--hidden_units',     action = 'store', type = int,   default = 4096)
argument_parser.add_argument('--epochs',           action = 'store', type = int,   default = 5)
argument_parser.add_argument('--gpu',              action = 'store', type = str,   default = 'gpu')
argument_parser.add_argument('--train_batch_size', action = 'store', type = int,   default = 128)
argument_parser.add_argument('--test_batch_size',  action = 'store', type = int,   default = 64)
argument_parser.add_argument('--valid_batch_size', action = 'store', type = int,   default = 64)

args = argument_parser.parse_args()

data_dir                  = args.data_dir
save_dir                  = args.save_dir
arch                      = args.arch
learning_rate             = args.learning_rate
hidden_units              = args.hidden_units
epochs                    = args.epochs
gpu                       = args.gpu
dataset_train_batch_size  = args.train_batch_size
dataset_valid_batch_size  = args.valid_batch_size
dataset_test_batch_size   = args.test_batch_size

device = torch.device("cuda:0" if torch.cuda.is_available() and gpu == 'gpu' else "cpu") # use cuda if available and wanted

#defining all global variables
global_data_dir = data_dir
global_train_dir = global_data_dir + '/train'
global_valid_dir = global_data_dir + '/valid'
global_test_dir = global_data_dir + '/test'

global_transforms_random_rotation = 20
global_transforms_resize = 256
global_transforms_center_crop = 224
global_transforms_random_resized_crop = 224

global_normalize_mean = [0.485, 0.456, 0.406]
global_normalize_standard = [0.229, 0.224, 0.225]

global_model_optimizer_learning_rate = learning_rate

global_training_epochs = epochs
global_training_print_every = 10

global_classifier_settings = OrderedDict([
    ('fc1', nn.Linear(25088, hidden_units)),
    ('relu1', nn.ReLU()),
    ("d_drop_out1", nn.Dropout(0.5)),
    ('fc2', nn.Linear(hidden_units, 1024)),
    ('relu2', nn.ReLU()),
    ("d_drop_out2", nn.Dropout(0.5)),
    ('fc3', nn.Linear(1024, 102)),
    ('output', nn.LogSoftmax(dim = 1)) #turn outputted numbers into probs for each label
])

#function to test model
def model_test(model, data_loader, criterion):

    test_loss = 0
    test_accuracy = 0

    for images, labels in data_loader:

        images = images.to(device)
        labels = labels.to(device)

        output = model.forward(images)

        test_loss += criterion(output, labels).item()

        probability = torch.exp(output)
        equality = (labels.data == probability.max(dim = 1)[1])
        test_accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, test_accuracy



#main
def main():
    #define transforms for the training, validation, and testing sets
    data_trans_train = transforms.Compose([transforms.RandomResizedCrop(global_transforms_random_resized_crop),
                                           transforms.RandomRotation(global_transforms_random_rotation),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(), #convert to tensor
                                           transforms.Normalize(global_normalize_mean, #means
                                                                global_normalize_standard)]) #standard deviation

    data_trans_valid = transforms.Compose([transforms.Resize(global_transforms_resize),
                                           transforms.CenterCrop(global_transforms_center_crop),
                                           transforms.ToTensor(), #convert to tensor
                                           transforms.Normalize(global_normalize_mean, #means
                                                                global_normalize_standard)]) #standard deviation

    data_trans_test = transforms.Compose([transforms.Resize(global_transforms_resize),
                                           transforms.CenterCrop(global_transforms_center_crop),
                                           transforms.ToTensor(), #convert to tensor
                                           transforms.Normalize(global_normalize_mean, #means
                                                                global_normalize_standard)]) #standard deviation

    #load the datasets with ImageFolder
    dataset_train = datasets.ImageFolder(global_train_dir, transform = data_trans_train)
    dataset_valid = datasets.ImageFolder(global_valid_dir, transform= data_trans_valid)
    dataset_test =  datasets.ImageFolder(global_test_dir, transform = data_trans_test)

    #using the image datasets and the trainforms, define the dataloaders
    data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size = dataset_train_batch_size, shuffle = True)
    data_loader_valid = torch.utils.data.DataLoader(dataset_valid, batch_size = dataset_valid_batch_size)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size = dataset_test_batch_size)

    #building
    if arch == 'densenet121':
        model = models.densenet121(pretrained = True) #load pretrained neural netwwork
    else:
        model = models.vgg16(pretrained = True) #load pretrained neural netwwork
        
    for param in model.parameters(): #do not train convolutional layer
        param.requires_grad = False  

    model.classifier = nn.Sequential(global_classifier_settings)
        
    criterion = nn.NLLLoss() #evaluate amount of model error
    optimizer = optim.Adam(model.classifier.parameters(), global_model_optimizer_learning_rate)    

    model.to(device)   

    #training
    steps = 0;

    for epoch in range(global_training_epochs):
        model.train() 
        running_loss = 0
        
        for images, labels in data_loader_train:
           
            images = images.to(device)
            labels = labels.to(device)
            
            steps += 1

            optimizer.zero_grad() #clearing weights
            
            output = model.forward(images) #getting output by forwarding the images
            
            loss = criterion(output, labels)
            
            loss.backward() #calculate with back-propagation

            optimizer.step() #adjusting the weights

            running_loss += loss.item()

            if steps % global_training_print_every == 0:
                model.eval()
                with torch.no_grad():
                    test_loss, test_accuracy = model_test(model, data_loader_valid, criterion)

                print(
                    f"TS: {datetime.now()} - "
                    f"Epoch: {epoch + 1} of {global_training_epochs} - "
                    f"Training loss: {running_loss / global_training_print_every:.3f} - "
                    f"Test loss: {test_loss / len(data_loader_valid):.3f} - "
                    f"Test accuracy: {test_accuracy / len(data_loader_valid) * 100:.3f}%"
                )
                running_loss = 0

                model.train() 


    #saving to checkpoint
    model.class_to_idx = dataset_train.class_to_idx

    print ("Model before saving:")
    print(model)

    torch.save({'arch' : arch,
                'learning_rate' : learning_rate,
                'gpu' : gpu,      
                'class_to_idx' : model.class_to_idx,
                'classifier' : model.classifier,
                'state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),    
                'idx_to_class': {v: k for k, v in dataset_train.class_to_idx.items()}, #from mentor                
               },
               save_dir)

if __name__ == '__main__':
    main()