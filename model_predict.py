#import required libraries and modules
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import random
from CNN_Model import CNNModel

#predict using saved model
#load the pre-trained CNN model, and use the same to make predictions on the input data.
def predict(in_data):

  reshape_in_data = in_data.values.reshape(1,-1)
  model = keras.models.load_model(r'C:\study\2ndSem\CI\computational-intelligence')
  predict_y=model.predict(reshape_in_data)
  classes_y=np.argmax(predict_y,axis=1)
  n_slice = 62
  
  peak = classes_y[0] * n_slice
  adjust = peak + 50
  return adjust
  
def main():
  dataset = r'C:\study\2ndSem\CI\computational-intelligence\dataset\Data_for_ML_Summer_2023.xlsx'
  #test_dataset = r'C:\study\2ndSem\CI_prev\dataset\T_Wand_000.xlsx' #test data with 2 index (or rows)
  print('Reading dataset: ', dataset)
  obj = CNNModel(dataset)
  print('Reducing noise and labelling data...')
  x_data = obj.reduce_noise_and_label()
  print('Grouping labelled data...')
  y_data = obj.group_labeled_data()
  xtrain, xtest, ytrain, ytest=train_test_split(x_data, y_data, test_size=0.25)
  peak = predict(x_data.iloc[155,0:])
  print('Peak: ',peak)

  

def check_GPUs():
  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Call main function
if __name__=="__main__":
  check_GPUs()
  main()