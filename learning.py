import utils as u
import predict as p
from sklearn.neural_network import MLPClassifier


def learn_model(x_train_features, y_train, x_val_features, y_val):
    classifier = MLPClassifier(solver='sgd', hidden_layer_sizes=88, max_iter=1, warm_start=True, batch_size=40)
    record = 0.925
    record_index = 0
    for i in range(1, 500):
        classifier.fit(x_train_features, y_train)
        W, b, classes = classifier.coefs_, classifier.intercepts_, classifier.classes_
        result_y = p.predict_with_model_params(x_val_features, W, b, classes)
        result = u.check_prediction(result_y, y_val)
        if result > record:
            record = result
            record_index = i
            u.save_data(
                {
                    'W': W,
                    'b': b,
                    'classes': classes
                }, 'models/' + str(i) + '_' + str(result) + '.pkl')
            print('\n', i, result, sep='---')
        else:
            print(i, result, sep='_', end=" ", flush=True)
    print(record_index, '=', record)


data = u.read_file('plik.pkl')
parting = 21601
learn_model(data[0][:parting], data[1][:parting].ravel(), data[0][parting:], data[1][parting:])
