from utils.lr_utils import load_dataset
from model.LogisticRegression import LogisticRegression
import numpy as np

train_dataset_path = 'datasets/train_catvnoncat.h5'
test_dataset_path = 'datasets/test_catvnoncat.h5'

# load test and train data
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset(
    train_dataset_path=train_dataset_path, test_dataset_path=test_dataset_path)

# format train and test data
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# normalize data
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

regression_model = LogisticRegression(train_set_x, train_set_y, test_set_x,
                                      test_set_y, classes, num_iterations=2000, learning_rate=0.005, print_cost=True)

# Train model
regression_model.optimize()

# Test model accuracy
Y_prediction_train = regression_model.predict(train_set_x)
Y_prediction_test = regression_model.predict(test_set_x)

print('Accuracy for train dataset: ', 100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100, "%")
print('Accuracy for test dataset: ', 100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100, "%")