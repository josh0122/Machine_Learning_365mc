import os 
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
data = pd.read_csv('C:\\Users\\365mc\\Desktop\\jsh_code\\yolov4_csv\\real_data.csv')    


# 타겟 변수의 형변환
mapping_dict = {'else' : 1,
                'lovehandle' : 2,
                'namsan' : 3,}
                # 'Class_4' : 4,
                # 'Class_5' : 5,
                # 'Class_6' : 6,
                # 'Class_7' : 7,
                # 'Class_8' : 8,
                # 'Class_9' : 9,}
after_mapping_target = data['label'].apply(lambda x : mapping_dict[x])


X = data[['front1','front2','front3','side1','side2','side3']]
y = after_mapping_target

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 0)

           
param_grid = {
    'n_estimators': [50,55,60,70,75,80,85,90,95,100,105,110],
    'max_depth' : [3,4,5,6,7,8,9,10],
    'min_samples_split': [1,2,3,4,5],
    'min_samples_leaf' : [1,2,3,4,5],

}
estimator = RandomForestClassifier()

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


grid_search.fit(train_x, train_y)


print(grid_search.best_params_)