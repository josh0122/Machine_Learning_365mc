# DecisionTree

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from datetime import datetime 
from dateutil import parser
from scipy.stats import ttest_1samp
from scipy.stats import ttest_ind
import matplotlib
from matplotlib import font_manager, rc
import platform
import os
from scipy.stats.mstats_basic import plotting_positions
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import tree

#data load
data =  pd.read_csv('C:\\Users\\365mc\\Desktop\\jsh_code\\yolov4_csv\\real_data.csv')
data=data.drop("Unnamed: 0",axis=1)

feature_cols = ['front1', 'front2', 'front3', 'side1', 'side2', 'side3']

X = data[feature_cols]
Y = data.label


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3 , random_state=1)

for i in range(1,11):
    treeclf = DecisionTreeClassifier(max_depth=i , random_state=1)
    treeclf = treeclf.fit(X_train,Y_train)
    Y_pred = treeclf.predict(X_test)
    print("Accuracy when depth is " , i         ,metrics.accuracy_score(Y_test,Y_pred))



treeclf1 = DecisionTreeClassifier(max_depth=5 , random_state=1)
treeclf1.fit(X,Y)


treeclf1_label= ['else_front','lovehandel','else_side','low','high','namsan']
with open("data.dot", 'w') as f:
   f = export_graphviz(treeclf1, out_file=f , feature_names = feature_cols, class_names=treeclf1_label)
os.environ["PATH"] += os.pathsep + 'C:\\Users\\365mc\\Desktop\\jsh_code\\'
os.system('dot -Tpng data.dot -o Decisiontree.png')


print("accuracy : " , metrics.accuracy_score(Y_test, Y_pred))

