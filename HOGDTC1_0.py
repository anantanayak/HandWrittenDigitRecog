import os,cv2 
import numpy as np
from sklearn import svm
from sklearn import metrics
from skimage.feature import hog
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
import time

#TRAINING DATASET 
path = "devnagriData/train"
imgsTrain=[]
labelsTrain=[]

#Extracting the train images and labels and storing in list
for (dirname, dirs, files) in os.walk(path):
        for filename in files:
            if filename.endswith('.jpg'):        
                img = cv2.imread(os.path.abspath(dirname+'/'+filename))        
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)               #Colour image to grayscale image 
                fd1, hog_image = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), 
                cells_per_block=(2, 2), visualize=True, multichannel=False)
                imgsTrain.append(fd1)
                labelsTrain.append(int(os.path.abspath(dirname)[-1]))
            else:
                pass
           
        
#Extracting the test images and labels and storing in lsit
path = "devnagriData/test"
imgsTest=[]
labelsTest=[]

for (dirname, dirs, files) in os.walk(path):
        for filename in files:
            if filename.endswith('.jpg'):
                img = cv2.imread(os.path.abspath(dirname+'/'+filename))
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)               #Colour image to grayscale image
                fd2, hog_image = hog(img_gray, orientations=9, pixels_per_cell=(8, 8), 
                cells_per_block=(2, 2), visualize=True, multichannel=False)
                imgsTest.append(fd2)

                labelsTest.append(int(os.path.abspath(dirname)[-1]))
            else:
                pass   
            
#Converting the list to array          
imgsTestN=np.array(imgsTest)
imgsTrainN=np.array(imgsTrain)


##Using SVM to train the model
#start = time.time()
#model = svm.SVC(kernel="linear")
#model.fit(imgsTrainN,labelsTrain)
#end = time.time()

#Using Decision Tree Classifier to train the model
start = time.time()
clf = DecisionTreeClassifier()
clf = clf.fit(imgsTrainN,labelsTrain)
end = time.time()

#Testing the model with Test images
predict = clf.predict(imgsTestN)

#Calculating the time required to train the model
print(end - start," seconds for training the model.")

#Finding out the accuracy of the model 
print("Model score is ",100*metrics.accuracy_score(labelsTest,predict),"%")
print("  ")

results=confusion_matrix(labelsTest,predict)
print(results)

y_actual=labelsTest
y_hat=predict

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)
    
print("  ")    
TP, FP, TN, FN = perf_measure(y_actual, y_hat)
print("True Positive=", TP)
print("False Positive=", FP)
print("True Negative=", TN)
print("False Negative=", FN)

Sensitivity = round((TP)/ (TP+FN),4)
Specificity = round((TN) / (FP + TN),4)
PPV = round(TP/(TP+FP),4)
ACC = round((TP+TN)/(TP+FP+FN+TN),4)

print("  ")
print("Sensitivity=", Sensitivity)
print("Specificity=", Specificity)
print("Precision=", PPV)
print("Accuracy=", ACC)
    


    
