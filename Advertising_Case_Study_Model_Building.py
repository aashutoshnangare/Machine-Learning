import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , r2_score

def MarvellousAdvertise(Datapath):
    Border = "-"*40

    #--------------------------------------------------
    #Step 1 : Load Dataaset
    #--------------------------------------------------

    print(Border)
    print("Step 1 : Load Dataset")
    print(Border)

    df = pd.read_csv(Datapath)

    print("Few records from the dataset :")
    print(df.head())

    #--------------------------------------------------
    #Step 2 : Remove Unwanted Columns
    #--------------------------------------------------

    print(Border)
    print("Step 2 : Remove Unwanted Column ")
    print(Border)

    print("Shape of Dataset before Removal",df.shape)

    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)

    print("Shape of Dataset after Removal",df.shape)

    print(Border)
    print("Clean Dataset is : ")
    print(Border)

    print(df.head())

    #--------------------------------------------------
    #Step 3 : Check missing values
    #--------------------------------------------------

    print(Border)
    print("Step 3 : Check Missing Values : ")
    print(Border)

    print("Missing values count : \n",df.isnull().sum())

    #--------------------------------------------------
    #Step 4 : Display Statistical Summary
    #--------------------------------------------------

    print(Border)
    print("Step 4 : Display Statistical Summary : ")
    print(Border)

    print(df.describe())

    #--------------------------------------------------
    #Step 5 : Correlation between Columns
    #--------------------------------------------------

    print(Border)
    print("Step 5 : Correlation Between Columns ")
    print(Border)

    print("Correlation matrix")
    print(df.corr())

    #--------------------------------------------------
    #Step 6 : Split Dataset Into Independent & Dependent Variables
    #--------------------------------------------------

    print(Border)
    print("Step 6 : Split Dataset Into Independent & Dependent Variables ")
    print(Border)

    X = df[['TV','radio','newspaper']] 
    Y = df['sales']

    print("Shape of Indepndent Variables",X.shape)
    print("Shape of Depndent Variables",Y.shape)

    #--------------------------------------------------
    #Step 7 : Split Dataset For Traning & Testing
    #--------------------------------------------------

    print(Border)
    print("Step 7 : Split Dataset For Traning & Testing")
    print(Border)

    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2 , random_state = 42)

    print("X_train shape : ",X_train.shape)
    print("X_test shape : ",X_test.shape)
    print("Y_train shape : ",Y_train.shape)
    print("Y_test shape : ",Y_test.shape)

    #--------------------------------------------------
    #Step 8 : Create The Model & Train The Model
    #--------------------------------------------------

    print(Border)
    print("Step 8 : Create The Model & Train The Model")
    print(Border)

    model = LinearRegression()

    model.fit(X_train,Y_train)

    #--------------------------------------------------
    #Step 9 :  Test The Model
    #--------------------------------------------------

    print(Border)
    print("Step 9 : Test The Model")
    print(Border)

    Y_pred = model.predict(X_test)

    #--------------------------------------------------
    #Step 10 :  Evaluate The Model
    #--------------------------------------------------

    print(Border)
    print("Step 10 : Evaluate The Model")
    print(Border)

    MSE = mean_squared_error(Y_test,Y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(Y_test,Y_pred)

    print("Mean Squared Error : ",MSE)
    print("Root Mean Squared Error : ",RMSE)
    print("R Square value : ",R2)

    #--------------------------------------------------
    #Step 11 : Calculate Model Co-efficient
    #--------------------------------------------------

    print(Border)
    print("Step 11 : Calculate Model Co-efficient")
    print(Border)

    for column , value in zip(X.columns,model.coef_):
        print(f"{column} : {value}")

    print("Intercept : ",model.intercept_)

    #--------------------------------------------------
    #Step 12 : Compare The Actual & Predicted Values
    #--------------------------------------------------

    print(Border)
    print("Step 12 : Compare The Actual & Predicted Values")
    print(Border)

    Result = pd.DataFrame({
        'Actual sale ' : Y_test.values,
        'Predicted sale' : Y_pred 
        })
    
    print(Result.head())

    


def main():
    MarvellousAdvertise("Advertising.csv")
    
if __name__ == "__main__":
    main()