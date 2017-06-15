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
    reshaped_x = list(map(lambda xi: xi.reshape(56, 56), x))
    hogged_x = list(map(lambda xi: hog(xi).flatten(), reshaped_x))
    return predict_after_extraction(hogged_x)


def hog(image):
    nwin_x = 5
    nwin_y = 5
    B = 7
    (L, C) = np.shape(image)
    H = np.zeros(shape=(nwin_x * nwin_y * B, 1))
    if C is 1:
        raise NotImplementedError
    step_x = np.floor(C / (nwin_x + 1))
    step_y = np.floor(L / (nwin_y + 1))
    cont = 0
    hxy = np.array([1, 0, -1])
    grad_xr = np.convolve(image.flatten(), hxy, mode='same').reshape(56, 56)
    grad_yu = np.convolve(image.T.flatten(), hxy, mode='same').reshape(56, 56).T
    angles = np.arctan2(grad_yu, grad_xr)
    magnit = np.sqrt((grad_yu ** 2 + grad_xr ** 2))
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


def predict_after_extraction(x_features):
    model = read_file('model.pkl')
    return predict_with_model_params(x_features, model['W'], model['b'], model['classes'])


def read_file(file_name):
    with open(file_name, 'rb') as f:
        return pkl.load(f)


def predict_with_model_params(x_features, W, b, classes):
    def predict_for_single(xi):
        steps = len(W) - 1
        for i in range(steps):
            xi = relu(W[i].T @ xi + b[i])
        xi = W[steps].T @ xi + b[steps]
        return classes[np.argmax(xi)]

    result = list(map(lambda xi: [predict_for_single(xi)], x_features))
    return np.array(result)


def relu(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X
