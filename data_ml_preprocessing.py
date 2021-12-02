import numpy as np
import pandas as pd


#data load
data =  pd.read_csv('C:\\Users\\365mc\\Desktop\\jsh_code\\yolov4_csv\\data.csv')
data=data.drop("Unnamed: 0",axis=1)

#value > 0 추출 
data= data.loc[(data['front1'] > 0) & (data['front2']>0) & (data['front3']>0) | (data['side1'] > 0) & (data['side2'] > 0) & (data['side3'] > 0)]

data.to_csv('real_data.csv')