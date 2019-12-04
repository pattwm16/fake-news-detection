from preprocessing import get_data
from text_feature_extractor import text_extract

# path to training dataset
train_path = "liar_dataset/train.tsv"
test_path = "liar_dataset/test.tsv"
valid_path = "liar_dataset/valid.tsv"

x_train, x_test, y_train, y_test = get_data(train_path, test_path, valid_path)

