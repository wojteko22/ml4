import main as m
import predict as p


def celina():
    data = m.load_main_data()
    x = data[0][:50]
    y = data[1][:50]
    result = p.predict(x)
    print(check_prediction(result, y))

def check_prediction(result_to_chech, y_true):
    """

    :param result_to_chech:
    :param y_true:
    :return:
    """
    error_count = 0
    for i in range(50):
        if result_to_chech[i][0] == y_true[i][0]:
            error_count += 1

    return error_count * 1.0 / 50

celina()