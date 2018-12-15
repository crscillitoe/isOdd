from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import to_categorical
from keras.regularizers import l1, l2, l1_l2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
import cv2
import numpy as np
import time
import skvideo.io

def main():
    # We sill save our generated model to this file
    model_file = 'is_odd_model.h5'

    # Define training data
    X = np.array([list(map(int, list(bin(int(x))[2:].zfill(100)))) for x in range(8000, 40000)])
    y = np.array([(x % 2, (x + 1) % 2) for x in range(8000, 40000)]) # Split data up into training and validation data.
    split_train = StratifiedShuffleSplit(n_splits=3, test_size = 0.4, train_size=0.6)
    for train_index, test_index in split_train.split(X, y):
        X_val, X_train = X[test_index], X[train_index]
        y_val, y_train = y[test_index], y[train_index]

    # Define model
    model = Sequential()

    # Input Layer
    model.add(Dense(1, input_shape=(100, ), kernel_initializer='he_normal'))

    # Hidden Layer
    model.add(Dense(20, activation='tanh', kernel_initializer='glorot_uniform',
                    bias_initializer='zeros', use_bias =True))

    # Output Layer
    model.add(Dense(2, kernel_initializer='he_normal'))
    model.add(Activation('sigmoid'))

    # Build the model
    model.compile(optimizer='sgd',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train,
              validation_data = (X_val, y_val),
              epochs=1000,
              batch_size=512)

    # Save the model to the output file
    model.save(model_file)

if __name__ == '__main__':
    main()
