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
import numpy as np
from keras.models import load_model

def main():
    model_file = 'is_odd_model.h5'
    model = load_model(model_file)

    for i in range(10):
        print(model.predict(np.array([i])))
    

if __name__ == '__main__':
    main()
