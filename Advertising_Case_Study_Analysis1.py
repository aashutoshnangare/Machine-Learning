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
    print("Step 5  : Correlation Between Columns ")
    print(Border)

    print("Correlation matrix")
    print(df.corr())


def main():
    MarvellousAdvertise("Advertising.csv")
    
if __name__ == "__main__":
    main()