import numpy as np
import utils as u
import predict as p
from sklearn.neural_network import MLPClassifier


def learn_model(x_train_features, y_train, x_val_features, y_val):
    classifier = MLPClassifier(solver='sgd', hidden_layer_sizes=(60, 50), random_state=1, max_iter=1,
                               # todo: zmienić warstwy
                               warm_start=True, batch_size=200)  # todo: zmienić batcha
    record = 0
    record_index = 0
    for i in range(1, 100):
        classifier.fit(x_train_features, y_train)
        W, b, classes = classifier.coefs_, classifier.intercepts_, classifier.classes_
        result_y = p.predict_with_model_params(x_val_features, W, b, classes)
        reshaped_result_y = np.array(list(map(lambda yi: [yi], result_y)))
        result = u.check_prediction(reshaped_result_y, y_val)
        if result > record:
            record = result
            record_index = i
            u.save_data(
                {
                    'W': W,
                    'b': b,
                    'classes': classes
                }, 'models/' + str(i) + '_' + str(result) + '.pkl')
            print(str(i) + ':\t', result)
    print(record_index, '=', record)


data = u.read_file('plik.pkl')
parting = 28000  # todo: zmienić
learn_model(data[0][:parting], data[1][:parting], data[0][parting:], data[1][parting:])
