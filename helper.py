from torchvision import transforms, models
import numpy as np
import torch
import json
from PIL import Image


# loads a checkpoint and rebuilds the model
def load_model_checkpoint(path):
    # load trained model from checkpoint
    state = torch.load(path)
    # first load pretrained model
    arch = state['arch']
    if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
        # assign trained classifier
        model.classifier = state['classifier']
    else:
        model = models.googlenet(pretrained=True)
        # assign trained classifier
        model.fc = state['classifier']
        
    # freeze model parameters 
    for param in model.parameters():
        param.requires_grad = False

    # assign optimizer hyperparameters
    model.state_dict = state['state_dict']
    # assign mapping of flower class values to the flower indices
    model.class_to_idx = state['class_to_idx']
    # how many epochs was used to train this model
    model.epochs = state['epochs']
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(225),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    pil_image = img_transforms(pil_image)
    return pil_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk, gpu, cat_to_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda" if gpu else "cpu")
    print('Using {} for training'.format(device.type))
    
    
    # Implement the code to predict the class from an image file
    with torch.no_grad():
        # Use cpu or gpu
        model.to(device)
        # no dropouts
        model.eval()
        # get tensor object of the image
        image = process_image(image_path)
        image = image.to(device) if gpu else image.type(torch.FloatTensor)
        image.to(device)
        # To resolve below issue 
        # Expected 4-dimensional input for 4-dimensional weight 64 3 3 3, 
        # but got 3-dimensional input of size [3, 224, 224] instead
        # convert 3D to 4D by adding dimension at 0 index
        image = image.unsqueeze(0)
        # forward on given image
        logps = model.forward(image)
        # get probabilities
        ps = torch.exp(logps)
        # get topk (5) probabilities and classes
        top_ps, top_class = ps.topk(topk, dim=1)
        idx_to_class = {idx:cls for cls, idx in model.class_to_idx.items()}
        # top_class.numpy() is 2D array - [[ 0 84 76 49 21]]
        top_c = top_class.cpu().numpy()[0] if gpu else top_class.numpy()[0]
        top_p = top_ps.cpu().numpy()[0] if gpu else top_ps.numpy()[0]

        classes = [idx_to_class[cls] for cls in top_c]
        flower_cat_names = [cat_to_name[c] for c in classes]
        return top_p, classes, flower_cat_names
    