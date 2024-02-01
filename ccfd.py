import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import warnings
import pickle
warnings.filterwarnings("ignore")

#def analyzeData(f="./creditcard.csv"):
   
df = pd.read_csv("./creditcard2.csv")
from sklearn.preprocessing import RobustScaler
new_df = df.copy()
new_df['Amount'] = RobustScaler().fit_transform(new_df['Amount'].to_numpy().reshape(-1, 1))
time = new_df['Time']
new_df['Time'] = (time - time.min()) / (time.max() - time.min())
print (new_df)
print(new_df)
#new_df =df.drop(['Time','Amount'],axis=1)



X=new_df.drop(columns=['Class'])
y=new_df['Class']
print(X,y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
 # # Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_trf = scaler.fit_transform(X_train)
X_test_trf = scaler.transform(X_test)
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import InputLayer,LeakyReLU,PReLU
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report
model = Sequential()

model.add(InputLayer(30))
model.add(Dense(100,activation='sigmoid'))
model.add(Dense(50,activation='sigmoid'))
model.add(Dense(25,activation='sigmoid'))
model.add(BatchNormalization())
model.add(Dense(1,activation='sigmoid'))
model.summary()
#checkpoint = ModelCheckpoint('model', save_best_only=True)
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(X_train,y_train,batch_size=50,epochs=20,verbose=1,validation_split=0.2)

model.save("./model/my_model.h5")

   
   
   



y_log = model.predict(X_test_trf)
y_pred = np.where(y_log>0.5,1,0)
print(y_pred)


  

    
    # Checking the accuracy
from sklearn.metrics import accuracy_score, confusion_matrix
print(classification_report(y_test, y_pred))
    
print("Accuracy Score : ", accuracy_score(y_test, y_pred))
print("Confusion Matrix: ", confusion_matrix(y_test,y_pred))


