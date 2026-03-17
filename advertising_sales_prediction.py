import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

def MarvellousAdvertise(DataPath):
    
    Border = "- "*50
    #----------------------------------------------------------------------------------
    # Step 1 : Load Dataset 
    #----------------------------------------------------------------------------------
    print(Border)
    print("Step 1 : Load Dataset")
    df  = pd.read_csv(DataPath)
    print(Border)
    print("Few records from the dataset : ")
    print(df.head())
    
    #----------------------------------------------------------------------------------
    # Step 2 : Remove Unwanted columns 
    #----------------------------------------------------------------------------------
    print(Border)
    print("Step 2 : Remove Unwanted columns ")
    print(Border)
    
    
    print("Shape of dataset before removal : ", df.shape)
    
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"],inplace=True)
    
    print("Shape of dataset after removal : ", df.shape)
    
    print(Border)
    print("Clean Dataset is : ")
    print(Border)
    
    print(df.head())
    
    
    #----------------------------------------------------------------------------------
    # Step 3 : Check Missing values  
    #----------------------------------------------------------------------------------
    print(Border)
    print("Step 3 : Check Missing values  ")
    print(Border)
    
    print("Missing values count : \n",df.isnull().sum())
    
    #----------------------------------------------------------------------------------
    # Step 4 : Display Statistical Summary  
    #----------------------------------------------------------------------------------
    print(Border)
    print("Step 4 : Display Statistical Summary ")
    print(Border)
    
    print(df.describe())
    
    #----------------------------------------------------------------------------------
    # Step 5 : Correlation between columns
    #----------------------------------------------------------------------------------
    print(Border)
    print("Step 5 : Correlation between columns ")
    print(Border)
    
    print("Correlation matrix")
    print(df.corr())
    
    #----------------------------------------------------------------------------------
    # Step 6 : Split Dataset into independent and Dependent Variables
    #----------------------------------------------------------------------------------
    print(Border)
    print("Step 6 : Split Dataset into independent and Dependent Variables")
    print(Border)
    
    X = df[["TV","radio","newspaper"]] 
    Y = df["sales"]
    
    print("Shape of Independent variables :",X.shape)
    print("Shape of Dependent variable ",Y.shape)
    
    #----------------------------------------------------------------------------------
    # Step 7 : Split Dataset for training and testing
    #----------------------------------------------------------------------------------
    print(Border)
    print("Step 7 : Split Dataset for training and testing")
    print(Border)
    
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state= 42)
    
    print("X_train  shape :",X_train.shape)
    print("X_test  shape :",X_test.shape)
    print("Y_train  shape :",Y_train.shape)
    print("Y_test  shape :",Y_test.shape)
    
    #----------------------------------------------------------------------------------
    # Step 8 : Create & train the model 
    #----------------------------------------------------------------------------------
    print(Border)
    print("Step 8 : Create & train the model ")
    print(Border)
    
    model = LinearRegression()
    
    model.fit(X_train,Y_train) # model Train using fit()
    
    #----------------------------------------------------------------------------------
    # Step 9 : test the model 
    #----------------------------------------------------------------------------------
    print(Border)
    print("Step 9 : test the model  ")
    print(Border)
    
    Ypred = model.predict(X_test) # Test the model
    
    
    #----------------------------------------------------------------------------------
    # Step 10 : Evaluate the model 
    #----------------------------------------------------------------------------------
    print(Border)
    print("Step 10 : Evaluate teh model ")
    print(Border)
    
    MSE = mean_squared_error(Y_test,Ypred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(Y_test,Ypred)
    
    print("Mean Squared Error :",MSE)
    print("Root Mean Squared Error : ", RMSE)
    print("R Square value : ", R2)
    
        
    #----------------------------------------------------------------------------------
    # Step 11 : Calculate model coefficient
    #----------------------------------------------------------------------------------
    print(Border)
    print("Step 11 : Calculate model coefficient ")
    print(Border)
    
    for column , value in zip(X.columns,model.coef_):
        print(f"{column} : {value}")
    
    print("Intercept : ", model.intercept_)
    
    #----------------------------------------------------------------------------------
    # Step 12 : Compare teh Actual and predicted values
    #----------------------------------------------------------------------------------
    print(Border)
    print("Step 12 : Compare teh Actual and predicted values")
    print(Border)
    
    Result = pd.DataFrame({
        'Actual sale': Y_test.values,
        'Predicted sale ': Ypred
        })
    
    print(Result.head())
    
    #----------------------------------------------------------------------------------
    # Step 13 : Actual Vs Predicted sales
    #----------------------------------------------------------------------------------
    print(Border)
    print("Step 13 : Actual Vs Predicted ")
    print(Border)
    
    plt.figure(figsize=(8,5))
    plt.scatter(Y_test,Ypred)
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual sales Vs Predicted sales")
    plt.grid(True)
    plt.show()
    
        
    

def main():
    
    # Entry point of program 
    csvFileName = "Advertising.csv"
    
    MarvellousAdvertise(csvFileName)
    
    
    

if __name__ == "__main__":
    main()