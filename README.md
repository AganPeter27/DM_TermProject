# DM_TermProject
Identifying and treating these diseases in a timely manner is critical for maintaining the health of the trees and maximizing the harvest. In this Project, we build a mango leaf disease classifier using Keras that can identify different types of mango leaf diseases based on images of the leaves

This project uses a convolutional neural network to classify mango leaves into one of four categories based on the type of disease they exhibit.
Dataset:
The dataset used in this project can be found on Kaggle https://www.kaggle.com/datasets/aryashah2k/mango-leaf-disease-dataset. It consists of 1000 images of mango leaves, with 250 images per class.
Requirements :
To run this project, you will need the following dependencies:
Python 3.x
TensorFlow 2.x
Keras
NumPy
Matplotlib
You can install these dependencies using pip:
pip install tensorflow keras numpy matplotlib
Usage :
Clone this repository to your local machine: git clone https://github.com/AganPeter27/DM_TermProject
Download the dataset and extract it to the data directory.
Run the train script to train the model
After training is complete, the model will be saved to the model directory.
Run the test script to evaluate the model on the test data
The script will output the test loss and accuracy, as well as overall accuracy on the test data
RESULT : After training the model for 50 epochs, we achieved a test accuracy of 87%. We also observed some overfitting, despite using data augmentation and dropout layers to prevent it. Further experiments with different hyperparameters and model architectures may be needed to improve the performance of the model.
