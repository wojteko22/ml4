import numpy as np
import main as m
from sklearn.neural_network import MLPClassifier


def learn_model(x_train, y_train, x_val, y_val):
    classifier = MLPClassifier(solver='sgd', hidden_layer_sizes=(45, 50), random_state=1, max_iter=1,
                               # todo: zmienić warstwy
                               warm_start=True, batch_size=200)  # todo: zmienić batcha
    for i in range(582):
        classifier.fit(x_train, y_train)
        result_y = classifier.predict(x_val)
        reshaped_result_y = np.array(list(map(lambda yi: [yi], result_y)))
        result = m.check_prediction(reshaped_result_y, y_val)
        print(str(i + 1) + ':\t', result)
        save_model(classifier, 'models/' + str(i + 1) + '_' + str(result) + '.pkl')


def save_model(model, file_name):
    m.save_data(
        {
            'coefs': model.coefs_,
            'intercepts': model.intercepts_,
            'classes': model.classes_
        }, file_name)


data = m.load_data('plik.pkl')
parting = 28000  # todo: zmienić
x_train = data[0][:parting]
y_train = data[1][:parting]
x_val = data[0][parting:]
y_val = data[1][parting:]
learn_model(x_train, y_train, x_val, y_val)
