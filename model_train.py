#import required libraries and modules
from CNN_Model import CNNModel
import tensorflow as tf
from sklearn.model_selection import train_test_split

#used for training
def main():
  dataset = r'C:\study\2ndSem\CI_prev\dataset\Data_for_ML_Summer_2023.xlsx' #1st set of test data
  #dataset1 = r'C:\study\2ndSem\CI_prev\dataset\T_Wand_000.xlsx' #2nd set of test data
  print('Reading dataset: ', dataset)
  obj = CNNModel(dataset)
  print('Reducing noise and labelling data...')
  x_data = obj.reduce_noise_and_label()
  print('Grouping labelled data...')
  y_data = obj.group_labeled_data()
  xtrain, xtest, ytrain, ytest=train_test_split(x_data, y_data, test_size=0.25)
  print('Training Model...')
  obj.train_model(xtrain, xtest, ytrain, ytest)
  

def check_GPUs():
  print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Call main function
if __name__=="__main__":
  check_GPUs()
  main()