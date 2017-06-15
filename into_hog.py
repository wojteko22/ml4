import main as m

data = m.load_main_data()
x = data[0]
hog = m.get_hog_features(list(map(lambda i: i.reshape(56, 56), x)))
m.save_data((hog, data[1]), 'plik.pkl')
