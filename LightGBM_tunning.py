# lightGBM

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score,roc_curve, auc
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


data = pd.read_csv('C:\\Users\\365mc\\Desktop\\jsh_code\\yolov4_csv\\real_data.csv')


X = data[['front1','front2','front3','side1','side2','side3']]
y = data['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

param_grid = {
    'colsample_bytree': [0.5,0.55 , 0.6, 0.7],
    'learning_rate' : [0.1,0.05,0.01,0.005,0.001],
    'max_depth': [-1, 2, 3],
    'n_estimators': [10,15],
}

estimator = lgb.LGBMClassifier()


kf = KFold(random_state=0,
           n_splits=10,
           shuffle=True,
          )
grid_search = GridSearchCV(estimator=estimator, 
                           param_grid=param_grid, 
                           cv=kf, 
                           n_jobs=-1, 
                           verbose=2
                          )

grid_search.fit(X_train, y_train)


print(grid_search.best_params_)



#############################################


# #load model
# estimator = joblib.load('lgb.pkl')

