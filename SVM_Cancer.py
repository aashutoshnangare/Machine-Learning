from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#---------------------------------------------
#Step 1 : Load datset
#---------------------------------------------

data = load_breast_cancer()

X = data.data
Y = data.target

#---------------------------------------------
#Step 2 : Split datset
#---------------------------------------------

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

#---------------------------------------------
#Step 3 : Create SVM Model
#---------------------------------------------

model = SVC(kernel='rbf',C=1,gamma='scale')
#rbf = Radial Basis Function

#---------------------------------------------
#Step 4 : Train SVM Models
#---------------------------------------------

model.fit(X_train,Y_train)

#---------------------------------------------
#Step 5 : Testing of SVM Dataset
#---------------------------------------------

Y_pred = model.predict(X_test)

#---------------------------------------------------------
#Step 7 : Evaluate  model
#---------------------------------------------------------

print(accuracy_score(Y_pred,Y_test))