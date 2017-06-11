# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import pickle as pkl
import numpy as np
import main as base


def predict(x):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
    :param x: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """
    coefs, inter, classes, activation = base.load_model('modelek2.pkl') # activation wywalić, rel najlepszy
    # http://scikit-learn.org/stable/modules/neural_networks_supervised.html
    # 1.17.7
    # coefs[0] to ponoć W1, coefs[1], to W2 itd., można to w pętli dać, albo na stałe, tyle ile mamy warst sieci
    # intercepts[0] to ponoć b1, intercepts[1], to b2 itd., można to w pętli dać, albo na stałe, tyle ile mamy warst sieci
    # wynik ostateczny to classes[wynik_nieostateczny]

    # w sumie potrzebne funkcje są chyba pod spodem, a Wątroba kazał mi to samemu robić :P

    # coefs, inter
    for i in x:

        y= base.hog(x.reshape(56,56))

        # for co in coefs:



    pass



ACTIVATIONS = {'identity': lambda x: x, 'tanh': np.tanh, 'logistic': base.sigmoid,
               'relu': base.relu, 'softmax': base.softmax}


def transpose_all_elements_in_fst_level_ndarray(array):
    return np.array(list(map(lambda x: x.transpose(), array)))


def prepare_list_one_dimen_array(array):
    return np.array(list(map(lambda x: base.prepare_one_dimen_array(x), array)))


# noinspection PyTypeChecker
def simple_predict(x):
    x = base.prepare_one_dimen_array(x)
    for i in range(len(_weights)):
        x = _weights[i] @ x + _bias[i]
        if i != len(_weights) - 1:
            x = _activation_function(x)

    return _classes[np.argmax(x)]


def predict_all(x, model_file_name):
    global _weights
    global _bias
    global _classes
    global _activation_function

    _weights, _bias, _classes, _activation_function = base.load_model(model_file_name)
    _activation_function = ACTIVATIONS[_activation_function]

    _weights = transpose_all_elements_in_fst_level_ndarray(_weights)
    _bias = prepare_list_one_dimen_array(_bias)

    return np.array(list(map(lambda i: np.array([simple_predict(i)]), x)))