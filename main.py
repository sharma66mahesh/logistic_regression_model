from utils.lr_utils import load_dataset

train_dataset_path = 'datasets/train_catvnoncat.h5'
test_dataset_path = 'datasets/test_catvnoncat.h5'

train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset(
    train_dataset_path=train_dataset_path, test_dataset_path=test_dataset_path)

print(train_set_x_orig)
print(train_set_y_orig)
print(test_set_x_orig)
print(test_set_y_orig)
print(classes)
