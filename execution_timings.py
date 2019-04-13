import time
import pickle
from data_description import test_data
from sklearn.preprocessing import scale

#Test data-
data = '/themes/modern/user_style.php?user_colors[bg_color]="</style><script>alert(411136083423)</script>'

#loading models
lgs = pickle.load(open('lgs_model', 'rb'))
svc = pickle.load(open('svm_model', 'rb'))

#Preproessing input data
vect = pickle.load(open('vect', 'rb'))
vect_data = vect.transform([str(data)])
scaled_X = scale(vect_data, with_mean=False)

vect_svm, X_test, y_test = test_data()
vect_svm_data = vect_svm.transform([str(data)])

#######SVM time############################
svm_time1 = time.time()

pred = svc.predict(vect_svm_data)  
    
svm_time2 = time.time()

print("Prediction time for SVM:", svm_time2-svm_time1)

##########Logistic Regression time###################
lg_time1 = time.time()
pred = lgs.predict(scaled_X)
lg_time2 = time.time()

print("Prediction time for LGR:", lg_time2-lg_time1)