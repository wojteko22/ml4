import numpy as np
import main as m
import predict as p
from sklearn.neural_network import MLPClassifier


def learn_model(x_train_features, y_train, x_val_features, y_val):
    classifier = MLPClassifier(solver='sgd', hidden_layer_sizes=(45, 50), random_state=1, max_iter=1,
                               # todo: zmienić warstwy
                               warm_start=True, batch_size=200)  # todo: zmienić batcha
    for i in range(582):
        classifier.fit(x_train_features, y_train)
        W, b, classes = classifier.coefs_, classifier.intercepts_, classifier.classes_
        result_y = p.predict_with_model_params(x_val_features, W, b, classes)
        reshaped_result_y = np.array(list(map(lambda yi: [yi], result_y)))
        result = m.check_prediction(reshaped_result_y, y_val)
        print(str(i + 1) + ':\t', result)
        m.save_data(
            {
                'W': W,
                'b': b,
                'classes': classes
            }, 'models/' + str(i + 1) + '_' + str(result) + '.pkl')


data = m.read_file('plik.pkl')
parting = 28000  # todo: zmienić
learn_model(data[0][:parting], data[1][:parting], data[0][parting:], data[1][parting:])
