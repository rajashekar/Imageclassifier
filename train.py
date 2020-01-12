import argparse
import os
import sys

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

from PIL import Image

parser = argparse.ArgumentParser(description='Image Classifier')

parser.add_argument('--data_dir', action='store', help='Data Directory', type=str, default=os.getcwd())
parser.add_argument('--save_dir', action='store', help='Save Directory', type=str, default=os.getcwd())
parser.add_argument('--arch', action='store', help='Pretrained vgg model', type=str, default='vgg19')
parser.add_argument('--learning_rate', action='store', help='Learning rate', type=float, default=0.0003)
parser.add_argument('--hidden_units', action='store', help='Hidden units', type=int, default=512)
parser.add_argument('--gpu', action='store_true', help='Enable GPU')

args = parser.parse_args()
data_dir = args.data_dir
save_dir = args.save_dir
arch = args.arch
hidden_units = args.hidden_units
learning_rate = args.learning_rate
gpu = args.gpu

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# Defining transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

non_train_transforms = transforms.Compose([transforms.Resize(225),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# Loading the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=non_train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=non_train_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True) 
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

# Build and train your network
if arch == 'vgg19':
    model = models.vgg19(pretrained=True)
else:
    print('No appropriate arch found. Exiting')
    sys.exit()

# Turn off gradients for our model - vgg19
for param in model.parameters():
    param.requires_grad = False

# There are 102 flower categories
num_labels = 102
classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(25088, 2*hidden_units)),
                            ('relu1', nn.ReLU()),
                            ('dropout1', nn.Dropout(p=0.2)),
                            ('fc2', nn.Linear(2*hidden_units, hidden_units)),
                            ('relu2', nn.ReLU()),
                            ('dropout2', nn.Dropout(p=0.2)),
                            ('fc3', nn.Linear(hidden_units, num_labels)),
                            ('output', nn.LogSoftmax(dim=1))
                           ]))

model.classifier = classifier

device = torch.device("cuda" if gpu else "cpu")
print('Using {} for training'.format(device.type))
# Using Negative Log Likelihood Loss - vgg19
criterion = nn.NLLLoss()
# Using optimization algorithm Adam with learning rate 
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
model.to(device)
epochs = 10
running_loss = 0
steps = 0
print_every = 10

print('Training the model.')
for epoch in range(epochs):
    for images, labels in trainloader:
        steps += 1
        # use if GPU available
        images, labels = images.to(device), labels.to(device)
        # set gradients of all model parameters to zero
        optimizer.zero_grad()
        # feed forward
        logps = model.forward(images)
        # get loss
        loss = criterion(logps, labels)
        # backward propagation
        loss.backward()
        # update weights
        optimizer.step()
        # calculate loss 
        running_loss += loss.item()
        
        # print every 10 batches
        if steps % print_every == 0:
            validation_loss = 0
            accuracy = 0

            #Turn off gradients
            with torch.no_grad():
                # set model to evaluation mode (dont use dropouts)
                model.eval()
                for images, labels in validloader:
                    # use GPU if available
                    images, labels = images.to(device), labels.to(device)
                    # Feed forward
                    logps = model.forward(images)
                    # get loss
                    loss = criterion(logps, labels)
                    validation_loss += loss.item()

                    # calculate accuracy
                    # getting probabilities by using exponential on log probabilities
                    ps = torch.exp(logps)
                    # using topk get top tuple with highest values 
                    top_ps, top_class = ps.topk(1, dim=1)
                    # how may of them are equal
                    equality = top_class == labels.view(*top_class.shape)
                    # calculate the mean to find accuracy of current epoch
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                print("Epoch: {}/{}...".format(epoch+1, epochs),
                      "Training loss: {:.3f}...".format(running_loss/len(trainloader)),
                      "Validation loss: {:.3f}...".format(validation_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                # for next epoch, make running loss to 0
                running_loss = 0
                # resume training (with dropouts)
                model.train()

# Do validation on the test set
print("Doing validation using test set.")
test_loss = 0
accuracy = 0
with torch.no_grad():
    model.eval()
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        logps = model.forward(images)
        loss = criterion(logps, labels)
        test_loss += loss.item()
        ps = torch.exp(logps)
        top_ps, top_class = ps.topk(1, dim=1)
        equality = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
    print("Test loss: {:.3f}...".format(validation_loss/len(testloader)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
    
# Save the checkpoint
path = save_dir + "/flowersvgg19gpu.pth"
state = {
    'classifier': classifier,
    'state_dict': optimizer.state_dict(),
    'class_to_idx': train_data.class_to_idx,
    'epochs': epochs,
}
print("Saving of trained model ...")
torch.save(state, path)
print("Saving of trained model is done..")
