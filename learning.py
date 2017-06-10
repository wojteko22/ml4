import numpy as np
import main as m
from sklearn.neural_network import MLPClassifier


def learn_model(x_train, y_train, x_val, y_val):
    """
    funkcja wyliczająca modele do mlc, sprawdza predykcję kolejno po każdej epoce
    zapisuje każdy plik z modelem oraz drukuje wynik predykcji
    :param x_train: ciąg z danymi wejściowymi do nauki modelu
    :param y_train: ciąg z danymi wyjściowymi do nauki modelu
    :param x_val: ciąg z danymi wejściowymi do walidacji
    :param y_val: ciąg z danymi wyjściowymi do walidacji
    :return: nic
    """
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(250, 250, 200), random_state=1, max_iter=1, #zmienić warstwy
                        activation='relu', warm_start=True, batch_size=20)  #zmienić batcha
    for i in range(100):
        clf.fit(x_train, y_train)
        predict = clf.predict(x_val)
        predict = np.array(list(map(lambda e: np.array([e]), predict)))
        print(i + 1, m.check_prediction(predict, y_val))
        m.save_model(clf, 'model\model_' + str(i + 1) + 'param(150_150_150)_relu_max_iter1_batch1.pkl')
        print('zapisano')


def bbb():
    data = m.load_main_data()
    x = data[0][:50]    #todo: wywalić 50
    hog = m.get_hog_features(list(map(lambda i: i.reshape(56, 56), x)))
    m.save_data((hog, data[1][:50]), 'plik.pkl') #todo: wywalić 50