# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import pickle as pkl
import numpy as np


def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """

    # for i in x:
    #     y= hog(x.reshape(56,56))

    def simple_predict(x):
        x = prepare_one_dimen_array(x)
        for i in range(len(W)):
            x = W[i] @ x + b[i]
            if i != len(coefs) - 1:
                x = relu(x)
        return classes[np.argmax(x)]

    x2 = get_hog_features(list(map(lambda i: i.reshape(56, 56), x)))
    coefs, intercepts, classes = load_model('modelek2.pkl')
    W = transpose_all_elements_in_fst_level_ndarray(coefs)
    b = prepare_list_one_dimen_array(intercepts)
    return np.array(list(map(lambda i: np.array([simple_predict(i)]), x2)))


def get_hog_features(x):
    return np.array(list(map(lambda y: hog(y).flatten(), x)))


def hog(image):
    nwin_x = 5
    nwin_y = 5  # podział obrazka
    B = 7  # liczba kierunków
    (L, C) = np.shape(image)
    H = np.zeros(shape=(nwin_x * nwin_y * B, 1))
    m = np.sqrt(L / 2.0)
    if C is 1:
        raise NotImplementedError
    step_x = np.floor(C / (nwin_x + 1))
    step_y = np.floor(L / (nwin_y + 1))
    cont = 0
    hx = np.array([[1, 0, -1]])
    hy = np.array([[-1], [0], [1]])
    hxy = np.array([1, 0, -1])
    grad_xr = np.convolve(image.flatten(), hxy, mode='same').reshape(56, 56)
    grad_yu = np.convolve(image.T.flatten(), hxy, mode='same').reshape(56, 56).T
    angles = np.arctan2(grad_yu, grad_xr)
    magnit = np.sqrt((grad_yu ** 2 + grad_xr ** 2))
    print('\n')
    for n in range(nwin_y):
        for m in range(nwin_x):
            cont += 1
            angles2 = angles[int(n * step_y):int((n + 2) * step_y), int(m * step_x):int((m + 2) * step_x)]
            magnit2 = magnit[int(n * step_y):int((n + 2) * step_y), int(m * step_x):int((m + 2) * step_x)]
            v_angles = angles2.ravel()
            v_magnit = magnit2.ravel()
            K = np.shape(v_angles)[0]
            bin = 0
            H2 = np.zeros(shape=(B, 1))
            for ang_lim in np.arange(start=-np.pi + 2 * np.pi / B, stop=np.pi + 2 * np.pi / B, step=2 * np.pi / B):
                for k in range(K):
                    if v_angles[k] < ang_lim:
                        v_angles[k] = 100
                        H2[bin] += v_magnit[k]
                bin += 1

            H2 = H2 / (np.linalg.norm(H2) + 0.01)
            H[(cont - 1) * B:cont * B] = H2
    return H


def transpose_all_elements_in_fst_level_ndarray(array):
    return np.array(list(map(lambda x: x.transpose(), array)))


def prepare_list_one_dimen_array(array):
    return np.array(list(map(lambda x: prepare_one_dimen_array(x), array)))


def prepare_one_dimen_array(x):
    return np.array(list(map(lambda y: np.array([y]), x)))


def relu(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X


def load_model(file_name):
    model = load_data(file_name)
    return model['coefs'], model['intercepts'], model['classes']


def load_data(file_name):
    with open(file_name, 'rb') as f:
        return pkl.load(f)