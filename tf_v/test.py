from utils import data_generator


X_train, X_valid, X_test = data_generator(dataset="Piano", framework="tf")
x = X_train[0]


