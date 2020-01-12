# Image Classifier
An image classifier built with PyTorch, this classifier can be trained to recognize different species of flowers. [This dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories is used to train this image classifier. 

## Pre-requisites
- Python 3
- Anaconda

## Getting Started
### Clone Repository
```
git clone https://github.com/rajashekar/Imageclassifier.git
cd Imageclassifier
```
### Training
To train data with gpu
```
python train.py <args>
```
Above command takes <br>
- --data_dir - type string - Place where train, validation & test data exists.
- --save_dir - type string - Place where to save model checkpoint
- --learning_rate - type float - Learning rate
- --hidden_units type int - Hidden units for network
- --gpu - To enable gpu or not

### Predict
To predict data
```
python predict.py <args>
```
- imagepath - type string - Image path.
- trained_model_path - type string - Path to saved model checkpoint.
- --top_k - type int - To print Top K most likely classes
- --category_names type string - Category names map file
- --gpu - To enable gpu or not

### Demo
Training data with gpu
```
python train.py --gpu

Using cuda for training
Training the model.
Epoch: 1/10... Training loss: 0.436... Validation loss: 4.167... Validation Accuracy: 0.127
Epoch: 1/10... Training loss: 0.399... Validation loss: 3.630... Validation Accuracy: 0.258
Epoch: 1/10... Training loss: 0.351... Validation loss: 3.153... Validation Accuracy: 0.269
Epoch: 1/10... Training loss: 0.325... Validation loss: 2.793... Validation Accuracy: 0.355
Epoch: 1/10... Training loss: 0.283... Validation loss: 2.334... Validation Accuracy: 0.447
Epoch: 1/10... Training loss: 0.260... Validation loss: 2.127... Validation Accuracy: 0.466
Epoch: 1/10... Training loss: 0.226... Validation loss: 1.805... Validation Accuracy: 0.551
Epoch: 1/10... Training loss: 0.228... Validation loss: 1.636... Validation Accuracy: 0.560
Epoch: 1/10... Training loss: 0.202... Validation loss: 1.531... Validation Accuracy: 0.593
Epoch: 1/10... Training loss: 0.203... Validation loss: 1.391... Validation Accuracy: 0.664
Epoch: 2/10... Training loss: 0.174... Validation loss: 1.305... Validation Accuracy: 0.671
Epoch: 2/10... Training loss: 0.161... Validation loss: 1.175... Validation Accuracy: 0.703
Epoch: 2/10... Training loss: 0.152... Validation loss: 1.125... Validation Accuracy: 0.708
Epoch: 2/10... Training loss: 0.145... Validation loss: 1.049... Validation Accuracy: 0.721
Epoch: 2/10... Training loss: 0.145... Validation loss: 1.001... Validation Accuracy: 0.746
Epoch: 2/10... Training loss: 0.146... Validation loss: 0.962... Validation Accuracy: 0.764
Epoch: 2/10... Training loss: 0.152... Validation loss: 1.037... Validation Accuracy: 0.719
Epoch: 2/10... Training loss: 0.131... Validation loss: 0.931... Validation Accuracy: 0.756
Epoch: 2/10... Training loss: 0.127... Validation loss: 0.848... Validation Accuracy: 0.772
Epoch: 2/10... Training loss: 0.123... Validation loss: 0.815... Validation Accuracy: 0.790
Epoch: 3/10... Training loss: 0.113... Validation loss: 0.797... Validation Accuracy: 0.773
Epoch: 3/10... Training loss: 0.112... Validation loss: 0.754... Validation Accuracy: 0.796
Epoch: 3/10... Training loss: 0.104... Validation loss: 0.664... Validation Accuracy: 0.828
Epoch: 3/10... Training loss: 0.108... Validation loss: 0.699... Validation Accuracy: 0.802
Epoch: 3/10... Training loss: 0.121... Validation loss: 0.760... Validation Accuracy: 0.804
Epoch: 3/10... Training loss: 0.107... Validation loss: 0.712... Validation Accuracy: 0.797
Epoch: 3/10... Training loss: 0.099... Validation loss: 0.703... Validation Accuracy: 0.805
Epoch: 3/10... Training loss: 0.104... Validation loss: 0.637... Validation Accuracy: 0.830
Epoch: 3/10... Training loss: 0.105... Validation loss: 0.575... Validation Accuracy: 0.847
Epoch: 3/10... Training loss: 0.096... Validation loss: 0.663... Validation Accuracy: 0.817
Epoch: 4/10... Training loss: 0.091... Validation loss: 0.663... Validation Accuracy: 0.830
Epoch: 4/10... Training loss: 0.090... Validation loss: 0.579... Validation Accuracy: 0.843
Epoch: 4/10... Training loss: 0.096... Validation loss: 0.596... Validation Accuracy: 0.842
Epoch: 4/10... Training loss: 0.087... Validation loss: 0.575... Validation Accuracy: 0.835
Epoch: 4/10... Training loss: 0.078... Validation loss: 0.558... Validation Accuracy: 0.842
Epoch: 4/10... Training loss: 0.081... Validation loss: 0.595... Validation Accuracy: 0.842
Epoch: 4/10... Training loss: 0.085... Validation loss: 0.529... Validation Accuracy: 0.851
Epoch: 4/10... Training loss: 0.083... Validation loss: 0.537... Validation Accuracy: 0.858
Epoch: 4/10... Training loss: 0.090... Validation loss: 0.520... Validation Accuracy: 0.859
Epoch: 4/10... Training loss: 0.083... Validation loss: 0.578... Validation Accuracy: 0.839
Epoch: 4/10... Training loss: 0.083... Validation loss: 0.526... Validation Accuracy: 0.849
Epoch: 5/10... Training loss: 0.089... Validation loss: 0.608... Validation Accuracy: 0.824
Epoch: 5/10... Training loss: 0.080... Validation loss: 0.498... Validation Accuracy: 0.868
Epoch: 5/10... Training loss: 0.085... Validation loss: 0.528... Validation Accuracy: 0.853
Epoch: 5/10... Training loss: 0.079... Validation loss: 0.553... Validation Accuracy: 0.842
Epoch: 5/10... Training loss: 0.077... Validation loss: 0.572... Validation Accuracy: 0.839
Epoch: 5/10... Training loss: 0.078... Validation loss: 0.458... Validation Accuracy: 0.866
Epoch: 5/10... Training loss: 0.079... Validation loss: 0.541... Validation Accuracy: 0.854
Epoch: 5/10... Training loss: 0.078... Validation loss: 0.535... Validation Accuracy: 0.851
Epoch: 5/10... Training loss: 0.073... Validation loss: 0.492... Validation Accuracy: 0.862
Epoch: 5/10... Training loss: 0.080... Validation loss: 0.533... Validation Accuracy: 0.853
Epoch: 6/10... Training loss: 0.081... Validation loss: 0.497... Validation Accuracy: 0.856
Epoch: 6/10... Training loss: 0.068... Validation loss: 0.492... Validation Accuracy: 0.855
Epoch: 6/10... Training loss: 0.072... Validation loss: 0.480... Validation Accuracy: 0.865
Epoch: 6/10... Training loss: 0.067... Validation loss: 0.473... Validation Accuracy: 0.869
Epoch: 6/10... Training loss: 0.073... Validation loss: 0.527... Validation Accuracy: 0.855
Epoch: 6/10... Training loss: 0.069... Validation loss: 0.450... Validation Accuracy: 0.866
Epoch: 6/10... Training loss: 0.077... Validation loss: 0.481... Validation Accuracy: 0.857
Epoch: 6/10... Training loss: 0.076... Validation loss: 0.471... Validation Accuracy: 0.870
Epoch: 6/10... Training loss: 0.067... Validation loss: 0.489... Validation Accuracy: 0.858
Epoch: 6/10... Training loss: 0.063... Validation loss: 0.430... Validation Accuracy: 0.874
Epoch: 7/10... Training loss: 0.072... Validation loss: 0.489... Validation Accuracy: 0.859
Epoch: 7/10... Training loss: 0.068... Validation loss: 0.485... Validation Accuracy: 0.853
Epoch: 7/10... Training loss: 0.065... Validation loss: 0.496... Validation Accuracy: 0.856
Epoch: 7/10... Training loss: 0.069... Validation loss: 0.485... Validation Accuracy: 0.853
Epoch: 7/10... Training loss: 0.066... Validation loss: 0.471... Validation Accuracy: 0.872
Epoch: 7/10... Training loss: 0.065... Validation loss: 0.427... Validation Accuracy: 0.880
Epoch: 7/10... Training loss: 0.065... Validation loss: 0.439... Validation Accuracy: 0.872
Epoch: 7/10... Training loss: 0.076... Validation loss: 0.428... Validation Accuracy: 0.876
Epoch: 7/10... Training loss: 0.061... Validation loss: 0.451... Validation Accuracy: 0.870
Epoch: 7/10... Training loss: 0.064... Validation loss: 0.427... Validation Accuracy: 0.876
Epoch: 7/10... Training loss: 0.073... Validation loss: 0.443... Validation Accuracy: 0.877
Epoch: 8/10... Training loss: 0.062... Validation loss: 0.388... Validation Accuracy: 0.885
Epoch: 8/10... Training loss: 0.054... Validation loss: 0.407... Validation Accuracy: 0.876
Epoch: 8/10... Training loss: 0.048... Validation loss: 0.406... Validation Accuracy: 0.879
Epoch: 8/10... Training loss: 0.073... Validation loss: 0.493... Validation Accuracy: 0.851
Epoch: 8/10... Training loss: 0.055... Validation loss: 0.408... Validation Accuracy: 0.878
Epoch: 8/10... Training loss: 0.060... Validation loss: 0.406... Validation Accuracy: 0.888
Epoch: 8/10... Training loss: 0.067... Validation loss: 0.404... Validation Accuracy: 0.876
Epoch: 8/10... Training loss: 0.060... Validation loss: 0.404... Validation Accuracy: 0.872
Epoch: 8/10... Training loss: 0.064... Validation loss: 0.410... Validation Accuracy: 0.888
Epoch: 8/10... Training loss: 0.069... Validation loss: 0.396... Validation Accuracy: 0.879
Epoch: 9/10... Training loss: 0.066... Validation loss: 0.428... Validation Accuracy: 0.869
Epoch: 9/10... Training loss: 0.055... Validation loss: 0.418... Validation Accuracy: 0.878
Epoch: 9/10... Training loss: 0.061... Validation loss: 0.420... Validation Accuracy: 0.867
Epoch: 9/10... Training loss: 0.058... Validation loss: 0.415... Validation Accuracy: 0.871
Epoch: 9/10... Training loss: 0.056... Validation loss: 0.378... Validation Accuracy: 0.894
Epoch: 9/10... Training loss: 0.063... Validation loss: 0.377... Validation Accuracy: 0.891
Epoch: 9/10... Training loss: 0.067... Validation loss: 0.402... Validation Accuracy: 0.880
Epoch: 9/10... Training loss: 0.058... Validation loss: 0.431... Validation Accuracy: 0.875
Epoch: 9/10... Training loss: 0.061... Validation loss: 0.412... Validation Accuracy: 0.886
Epoch: 9/10... Training loss: 0.059... Validation loss: 0.336... Validation Accuracy: 0.899
Epoch: 10/10... Training loss: 0.058... Validation loss: 0.397... Validation Accuracy: 0.883
Epoch: 10/10... Training loss: 0.053... Validation loss: 0.386... Validation Accuracy: 0.894
Epoch: 10/10... Training loss: 0.060... Validation loss: 0.385... Validation Accuracy: 0.887
Epoch: 10/10... Training loss: 0.052... Validation loss: 0.387... Validation Accuracy: 0.891
Epoch: 10/10... Training loss: 0.056... Validation loss: 0.381... Validation Accuracy: 0.890
Epoch: 10/10... Training loss: 0.046... Validation loss: 0.406... Validation Accuracy: 0.890
Epoch: 10/10... Training loss: 0.060... Validation loss: 0.400... Validation Accuracy: 0.892
Epoch: 10/10... Training loss: 0.058... Validation loss: 0.404... Validation Accuracy: 0.879
Epoch: 10/10... Training loss: 0.060... Validation loss: 0.439... Validation Accuracy: 0.880
Epoch: 10/10... Training loss: 0.059... Validation loss: 0.403... Validation Accuracy: 0.882
Epoch: 10/10... Training loss: 0.049... Validation loss: 0.414... Validation Accuracy: 0.887
Doing validation using test set.
Test loss: 0.414... Test Accuracy: 0.835
Saving of trained model ...
Saving of trained model is done..
```

Predicting with cpu
```
python predict.py flowers/test/1/image_06743.jpg flowersvgg19gpu.pth

Using cpu for training
Given actual Image flowers/test/1/image_06743.jpg belongs to flower category pink primrose
Top 5 classes : ['1', '83', '89', '86', '51'] with respective probabilities : [9.8114121e-01 1.4599325e-02 1.5717540e-03 7.0153625e-04 5.9656572e-04]
Model Prediction - Flower flowers/test/1/image_06743.jpg belongs to category pink primrose with probability 0.981141209602356
```

Predicting with gpu
```
python predict.py flowers/test/1/image_06743.jpg flowersvgg19gpu.pth --gpu

Using cuda for training
Given actual Image flowers/test/1/image_06743.jpg belongs to flower category pink primrose
Top 5 classes : ['1', '83', '89', '86', '51'] with respective probabilities : [9.8114091e-01 1.4599325e-02 1.5717547e-03 7.0153555e-04 5.9656572e-04]
Model Prediction - Flower flowers/test/1/image_06743.jpg belongs to category pink primrose with probability 0.9811409115791321
```
