
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score # Accuracy metrics
import pickle

db = pd.read_csv('test14.csv')
print(db.tail())
X = db.drop('class', axis=1)  # features
y = db['class']  # target value
# print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
print(X_train)
# #
pipelines = {

    'lr': make_pipeline(StandardScaler(), LogisticRegression()),
    'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),

}
a = pipelines.keys()
# print(a)
# #
fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model
    # print(fit_models)
    f = fit_models['lr'].predict(X_test)
    # print(f)
#
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    # print(algo, accuracy_score(y_test, yhat))
    f = fit_models['lr'].predict(X_test)
    print(f)

with open('lrmode_body_exercise.pkl', 'wb') as f:
    pickle.dump(fit_models['lr'], f)
