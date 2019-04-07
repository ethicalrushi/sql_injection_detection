import pickle
from data_description import vect
from sklearn.preprocessing import scale


vect = pickle.load(open('vect', 'rb'))
def compute(data):
    vect_data = vect.transform([str(data)])
    scaled_X = scale(vect_data, with_mean=False)
    svc = pickle.load(open('lgs_model', 'rb'))
    pred = svc.predict(scaled_X)
    print(pred)

    if int(pred)==0:
        return 'Normal Query'
    else:
        return 'SQL Injection Query'
