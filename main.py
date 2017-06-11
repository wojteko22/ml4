import numpy as np
import pickle as pkl


# def load_data():
#     with open('train.pkl', 'rb') as f:
#         return pkl.load(f)

def load_main_data():
    """
    ładuje początkowe dane treningowe z pliku
    :return: obiekt danych początkowych pliku z danymi
    """
    return load_data('train.pkl')


def load_data(file_name):
    """
    Odczytuje dane z pliku pickle
    :param file_name: nazwa pliku
    :return: obiekt z pliku
    """
    with open(file_name, 'rb') as f:
        return pkl.load(f)


def load_model(file_name):
    """
    Wczytuje model mlc z pliku
    :param file_name: nazwa pliku
    :return: ktorka z danymi modelu mlc
    """
    model = load_data(file_name)
    return model['coefs'], model['intercepts'], model['classes'], model['activation']


def save_data(data, file_name):
    """
    Zapisuje dane do pliku pickle
    :param data: obiekt
    :param file_name: nazwa pliku
    :return:
    """
    output = open(file_name, 'wb')
    pkl.dump(data, output)
    return


def save_model(model, file_name):
    """
    Zapisuje model mlc do pliku
    :param model: model mlc
    :param file_name: nazwa pliku
    :return:
    """
    save_data({'coefs': model.coefs_, 'intercepts': model.intercepts_, 'classes': model.classes_,
               'activation': model.activation}, file_name)


def softmax(x):
    """
    Funkcja obliczająca softmax
    :param x: tablica z danuli wejściowymi Nx1 lub N
    :return: ndarray z sofrmaxem Nx1 lub N
    """
    values = np.array(list(map(lambda i: np.exp(i), x)))
    val_sum = values.sum()
    return np.array(list(map(lambda i: i / val_sum, values)))


def sigmoid(x):
    """
    Oblicza funkcję sigmoid dla tablicy Nx1 lub N
    :param x: wektor wejsciowych wartosci Nx1 lub N
    :return: wektor wyjściowych wartości funkcji sigmoidalnej dla wejścia x, Nx1 lub N
    """
    return np.divide(1, np.add(1, np.exp(-x)))


def relu(x):
    """
    Oblicza relu dla tablicy Nx1 lub N
    :param x: wektor wejsciowych wartosci Nx1 lub N
    :return: wektor wyjściowych wartości funkcji relu dla wejścia x, Nx1 lub N
    """
    np.clip(x, 0, np.finfo(x.dtype).max, out=x)
    return x


def check_prediction(result_to_chech, y_true):
    """

    :param result_to_chech:
    :param y_true:
    :return:
    """
    error_count = 0
    for i in range(result_to_chech.shape[0]):
        if result_to_chech[i][0] == y_true[i][0]:
            error_count += 1

    return error_count * 1.0 / result_to_chech.shape[0]


def prepare_one_dimen_array(x):
    """
    Funkcja opakowuje listę, ndarray w w głąb
    [1, 2, 3] -> [[1], [2], [3]]
    :param x: lista/ndarray wejściowa Nx
    :return: lista/ndarray wyjściowa Nx1x
    """
    return np.array(list(map(lambda y: np.array([y]), x)))


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


def ania():
    data = load_main_data()
    x = data[0]
    hog = get_hog_features(list(map(lambda i: i.reshape(56, 56), x)))
    save_data((hog, data[1]), 'plik.pkl')
