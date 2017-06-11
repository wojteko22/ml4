import main as m
import predict as p


def celina():
    data = m.load_main_data()
    x = data[0][:50]
    y = data[1][:50]
    result = p.predict(x)
    print(m.check_prediction(result, y))


celina()