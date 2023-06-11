from sklearn.datasets import load_breast_cancer
import numpy as np

# Data Import
data = load_breast_cancer()
#print(data)

# Organizing Data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

#print(features)


# Data Splitting
from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(features, labels, test_size=0.2, random_state=22)

#print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)

# Model Training and Accuracy
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_absolute_error as mae

lr = LinearRegression()
rfc = RandomForestClassifier()
svc = SVC()
gnb = GaussianNB()

models = [lr, rfc, svc, gnb]

for model in range(4):
    m = models[model].fit(X_train, Y_train)
    pred = m.predict(X_val)
    print(f'{models[model]} : ')
    print(f'Validation Accuracy : {1-(mae(Y_val, pred))}')

# GausianNB is the best one
