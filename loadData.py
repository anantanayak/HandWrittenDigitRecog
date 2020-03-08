import os,cv2 
import matplotlib.pyplot as plt
import numpy as np
import csv 
import pandas as pd
import joblib
from sklearn import datasets
from sklearn import svm
from sklearn import metrics



path = "devnagriData/train"
imgsTrain=[]
labelsTrain=[]

for (dirname, dirs, files) in os.walk(path):
        for filename in files:
            if filename.endswith('.jpg'):        
                img = cv2.imread(os.path.abspath(dirname+'/'+filename))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_gray = cv2.GaussianBlur(img_gray, (15,15),0)
                imgsTrain.append(img_gray)
                labelsTrain.append(int(os.path.abspath(dirname)[-1]))
            else:
                pass
           
        
        
path = "devnagriData/test"
imgsTest=[]
labelsTest=[]

for (dirname, dirs, files) in os.walk(path):
        for filename in files:
            if filename.endswith('.jpg'):
                img = cv2.imread(os.path.abspath(dirname+'/'+filename))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_gray = cv2.GaussianBlur(img_gray, (15,15),0)
                imgsTest.append(img_gray)
                labelsTest.append(int(os.path.abspath(dirname)[-1]))
            else:
                pass        
            
imgsTestN=np.array(imgsTest)
imgsTrainN=np.array(imgsTrain)
train_dataset = imgsTrainN.reshape((17000,28*28))
test_dataset = imgsTestN.reshape((3000,28*28))     

#dup = []
#for k in train_dataset:
#    for i in k:
#        dup.append(i)
#
#print (max(dup), min(dup), np.median((dup)))

ret2,Bi_Train = cv2.threshold(train_dataset,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret3,Bi_Test = cv2.threshold(test_dataset,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

model = svm.SVC(kernel="linear")
model.fit(Bi_Train,labelsTrain)
#joblib.dump(model,"model/SVM")

predict = model.predict(Bi_Test)

print("Model score",100*metrics.accuracy_score(labelsTest,predict),"%")
