import os 
import pandas as pd 

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
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

clf = RandomForestClassifier(n_estimators=70, max_depth=5,random_state=0,oob_score = True)
clf.fit(train_x,train_y)

predict1 = clf.predict(test_x)
print(accuracy_score(test_y,predict1))


print("훈련 세트 정확도: {:.3f}".format(clf.score(train_x, train_y)) )
print("테스트 세트 정확도: {:.3f}".format(clf.score(test_x, test_y)) )
print("OOB 샘플의 정확도: {:.3f}".format(clf.oob_score_) )

# raw_data = {'front1': [1, 2, 3, 4],
#             'front2': [10, 20, 30, 40],
#             'front3': [100, 200, 300, 400],
#             'side1':  [0,0,0,0],
#             'side2':  [0,0,0,0],
#             'side3':  [0,0,0,0]}
# data = pd.DataFrame(raw_data)
# predict1 = clf.predict(data)
# print(predict1)