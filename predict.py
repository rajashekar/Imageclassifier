import argparse
import helper
import numpy as np
import json

parser = argparse.ArgumentParser(description='Image prediction')

parser.add_argument('imagepath', action='store', help='Image path', type=str)
parser.add_argument('trained_model_path', action='store', help='Trained model path', type=str)
parser.add_argument('--top_k', action='store', help='Top K most likely classes', type=int, default=5)
parser.add_argument('--category_names', action='store', help='Category names map file', type=str, default='cat_to_name.json')
parser.add_argument('--gpu', action='store_true', help='Enable GPU')

args = parser.parse_args()

imagepath = args.imagepath
trained_model_path = args.trained_model_path
gpu = args.gpu
category_names = args.category_names
top_k = args.top_k

with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

# load model checkpoint
model = helper.load_model_checkpoint(trained_model_path)
# predict using model
probs, classes, flower_cat_names = helper.predict(imagepath, model, top_k, gpu, cat_to_name)

np_probs = np.array(probs)
high_prob = np.amax(np_probs)
high_prob_idx = np.where(np_probs == np.amax(np_probs))
actual_flower_idx = imagepath.split('/')[-2]
actual_flower_cat = cat_to_name[actual_flower_idx]
print("Given actual Image {} belongs to flower category {}".format(imagepath, actual_flower_cat))
print("Top {} classes : {} with respective probabilities : {}".format(top_k,classes,probs))
print("Model Prediction - Flower {} belongs to category {} with probability {}".format(imagepath,flower_cat_names[high_prob_idx[0][0]],high_prob))