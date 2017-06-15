import utils as u
import predict as p


dimen = 101


data = u.read_train_file()
x = data[0][:dimen]
y = data[1][:dimen]
result = p.predict(x)
if result.shape != (dimen, 1):
    print("Error!")
print(u.check_prediction(result, y))
