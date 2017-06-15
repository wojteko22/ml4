import main as m
import predict as p

dimen = 50


def celina():
    data = m.load_main_data()
    x = data[0][:dimen]
    y = data[1][:dimen]
    result = p.predict(x)
    if result.shape != (dimen, 1):
        print("Error!")
    print(m.check_prediction(result, y))


celina()
