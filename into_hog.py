import utils as m
import predict as p

data = m.read_train_file()
x = data[0]
y = data[1]
m.save_data((p.hog_all(x), y), 'plik.pkl')
