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

def prediction(num, model):
    number = np.array(list(map(int, list(bin(int(num))[2:].zfill(100))))).reshape(1, 100)
    return model.predict(number)

def isOdd(num, model):
    return not bool(np.argmax(prediction(num, model)))


def main():
    model_file = 'is_odd_model.h5'
    model = load_model(model_file)

    for i in range(50000):
        if(i % 500 == 0) :
            print(f'Testing numbers {i} through {i + 500}...')
        if isOdd(i, model) != ((i % 2) == 1):
            print(f'Test {i} Failed!')
            return

    print(f'Tests passed. {model_file} is valid model.')

if __name__ == '__main__':
    main()
