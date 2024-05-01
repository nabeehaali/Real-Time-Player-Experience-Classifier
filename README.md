# Real-Time-Player-Experience-Classifier-from-Platformer-Games

In this paper, we present our methodology for leveraging the Facial Expression Recognition Challenge Dataset (FERC) to classify users' emotions in real-time. We describe the construction of a multilayered Convolutional Neural Network (CNN) designed to train and validate the dataset, achieving a noteworthy accuracy of 60\%. Subsequently, we applied this model to a custom-built program capable of receiving inputs through a webcam, enabling the real-time mapping of detected emotions to exaggerated expressions, dynamically altering in response to the user's emotional state.

## How to Run the Project
1. Download folder
2. Install dependencies listed on the top of each file (Model_CNN.py and Webcam_Test.py)
```
  import pandas as pd
  import numpy as np
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
  from keras.models import Sequential
  from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
  import matplotlib.pyplot as plt
  import tensorflow as tf

  import cv2
  from keras.models import load_model
```
4. Change file path in Model_CNN.py to load the [Training Set](https://drive.google.com/file/d/1O55u67my1RDNW7uyYJ0qoClh90NWYoRJ/view?usp=sharing) from google drive
6. Run Model_CNN.py (you should be able to view the loss and accuracy on both training and validation along with the confusion matrix)
7. Once the model is saved (CNN_Emotion_Model.h5), run Webcam_Test.py to test the model on live webcam footage

## Test the Game Implementation
We integrated our CNN model to dynamically adjust the difficulty of game levels of a simple 2D endless platformer game made in Unity. To run the game:
1. Download [AIEndlessRunner](https://drive.google.com/drive/folders/1GlgDSkbOpRTtNGN40ki7L3jfbHf8IIPq?usp=sharing) from google drive 
2. Open the project in Unity (version 2022.3.20f1)
3. Open the scene file
4. Run the scene by clicking play
