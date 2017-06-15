import numpy as np
import pickle as pkl
import predict as p


def load_main_data():
    return read_file('train.pkl')


def read_file(file_name):
    with open(file_name, 'rb') as f:
        return pkl.load(f)


def save_data(data, file_name):
    output = open(file_name, 'wb')
    pkl.dump(data, output)


def get_hog_features(x):
    return np.array(list(map(lambda y: p.hog(y).flatten(), x)))


def check_prediction(result_y, real_y):
    good_count = 0
    size = result_y.shape[0]
    for i in range(size):
        if result_y[i][0] == real_y[i][0]:
            good_count += 1
    return good_count * 1.0 / size
