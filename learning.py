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
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(45, 50), random_state=1, max_iter=1, #zmienić warstwy (ja policzyłem na razie dla tych 45, 50)
                        activation='relu', warm_start=True, batch_size=200)  #zmienić batcha (ja policzyłem na razie dla tego 200)
    for i in range(100):
        clf.fit(x_train, y_train)
        predict = clf.predict(x_val)
        predict = np.array(list(map(lambda e: np.array([e]), predict)))
        print(i + 1, m.check_prediction(predict, y_val))
        m.save_model(clf, 'model_' + str(i + 1) + 'param(150_150_150)_relu_max_iter1_batch1.pkl')
        print('zapisano')


def bogusia():
    """
    Bogusia Ci kurwa model liczy i przez to robi dobrze
    :return: nic kurwa nie zwraca - masz logi i pliki z modelami dla każdej epoki
    """
    data = m.load_data('plik.pkl')
    parting = 28000 # to chyba można zmienić
    x_train = data[0][:parting]
    y_train = data[1][:parting]
    x_val = data[0][parting:]
    y_val = data[1][parting:]
    learn_model(x_train, y_train, x_val, y_val)


bogusia()   # jak odpalicie Bogusię, to się Wam pliki pojawią i przewidywane wyniki, bierzecie najlepszy, a resztę wywalacie, tych plików możecie generować, ile chcecie, w końcu zaczną spadać wyniki
