# -*- coding: utf-8 -*-
"""Prediction2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KIjt2Sg6uVpJ3J-1kL8WbyhsRLFNoaLh
"""
import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import os
from io import StringIO
from datetime import datetime

def main():
    st.title("ABC Corp.")
    st.title("Automated Machine Learning Web")
    data_file='./DataDump/file'+datetime.now().strftime("%d%b%Y_%H%M%S%f")+'.csv'
    file_bytes = st.file_uploader("Upload a file")
    data_load_state = st.text("Upload your data")
    try:
        if file_bytes is not None:  
            with open(data_file, mode='w', newline='') as f:
                print(file_bytes.getvalue().strip('\r\n'), file=f)
                data_load_state.text("Upload....Done!")
            dataDF = pd.read_csv(data_file)
    except FileNotFoundError:
        st.error('File not found.')


    st.header("Data Exploration")

    X = ""
    y = ""
    X_train=''
    X_test=''
    y_train=''
    y_test = ''
    y_pred = ''

    @st.cache
    def load_data():
            data = pd.read_csv(data_file)
            # st.write(data.head())
            return data
              
    if st.checkbox("Show Data HEAD or TAIL"):
            select_option = st.radio("Select option", ['HEAD','TAIL'])
            if select_option == 'HEAD':
                    st.write(dataDF.head())
            elif select_option == "TAIL":
                    st.write(dataDF.tail())

    if st.checkbox("Show Full Data"):
            st.write(load_data())
            data_load_state.text("Loading data....Done!")

    if st.checkbox("Data Info"):
        st.text("Data Shape")
        st.write(load_data().shape)
        st.text("Data Columns")
        st.write(load_data().columns)
        st.text("Data Type")
        st.write(load_data().dtypes)
        st.text("Count of NaN values")
        st.write(load_data().isnull().any().sum())

    if st.checkbox("Select Target Column"):
            all_columns = load_data().columns
            target = st.selectbox("Select",all_columns)
            #st.write(load_data()[names])
            
    if st.checkbox("Auto Discard Columns"):
        for column in dataDF:
            if (dataDF[column].nunique() == dataDF.shape[0]):
                dataDF.drop([column], axis=1, inplace=True)
        for column in dataDF:
            if 'name' in column.lower():
                dataDF.drop([column], axis=1, inplace=True)
        
        st.text("Data Columns")
        st.write(dataDF.columns)
        st.text("Count of NaN values")
        st.write(dataDF.isnull().any().sum())

    if st.checkbox("Preprocess Object Type Columns"):
        obj_df = dataDF.select_dtypes(include=['object']).copy()
        dataDF=dataDF.select_dtypes(exclude=['object'])
        try:
          one_hot = pd.get_dummies(obj_df)#,drop_first=True)
        except:
          one_hot = pd.DataFrame()

        main_df = pd.concat([one_hot,dataDF],axis=1)


    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, accuracy_score

    sc = StandardScaler()
    st.header("Split DataSet into Train and Test")
    def normalize (x):
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        norm=scaler.fit(x)
        return(norm)
        
    if st.checkbox("Split"):
        X = main_df.drop([target],axis=1)
        #X = X.apply(normalize)
        y = main_df[target]
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.01,random_state=20)

    if st.checkbox("Normalize Columns"):
##        def normalize (x):
##            return ( (x-np.min(x))/ (max(x) - min(x)))
        norm=normalize(X_train)
        X_train = norm.transform(X_train)
        X_test = norm.transform(X_test)

##    if st.checkbox("Preform Standard Scale"):
##            X_train = sc.fit_transform(X_train)
##            X_test = sc.transform(X_test)

    if st.checkbox("Show X_test,X_train,y_test,y_train"):
            st.write("X_train")
            st.write(X_train)
            st.write(X_train.shape)
            st.write("X_test")
            st.write(X_test)
            st.write(X_test.shape)
            st.write("y_train")
            st.write(y_train)
            st.write(y_train.shape)
            st.write("y_test")
            st.write(y_test)
            st.write(y_test.shape)

    def gradBoost (X,y):
        from sklearn.ensemble import GradientBoostingClassifier
        gradientBoosting = GradientBoostingClassifier()
        gradientBoosting.fit(X,y)
        return (gradientBoosting)
    def randForest (X,y):
        from sklearn.ensemble import RandomForestClassifier
        randomForest = RandomForestClassifier()
        randomForest.fit(X,y)
        return(randomForest)
    def svm (X,y):
        from sklearn import svm
        clf = svm.SVC(kernel='linear')
        clf.fit(X,y)
        return(clf)
    def xgb (X,y):
        import xgboost as xgb
        xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 10)
        xg_reg.fit(X,y)
        return(xg_reg)
    def linearReg (X,y):
        from sklearn.linear_model import LinearRegression
        lineReg = LinearRegression().fit(X, y)
        return(lineReg)
    def lassoReg (X,y):
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=0.01)
        lasso.fit(X_train, y_train)
        return(lasso)
        

    if st.checkbox("ML Algorithms"):
        if (main_df[target].nunique() == 2):
            st.header("Using Binary Classification Algorithms")
            GB = gradBoost (X_train, y_train)
            st.write('Accuracy of Gradient Boosting classifier on test set: {:.2f}'.format(GB.score(X_test, y_test)))
            RF = randForest (X_train, y_train)
            st.write('Accuracy of Random Forest classifier on test set: {:.2f}'.format(RF.score(X_test, y_test)))
            SVM = svm (X_train, y_train)
            st.write('Accuracy of SVM classifier on test set: {:.2f}'.format(SVM.score(X_test, y_test)))
        elif (main_df[target].nunique() / main_df[target].count() < .1):
            st.header("Using Multi-Class Classification Algorithms")
            
        else:
            st.header("Using Regression Algorithms")
            from sklearn.metrics import mean_squared_error, r2_score
            LReg = linearReg (X_train, y_train)
            st.write('R-squared value for Linear Regression predictor on test set: {:.2f}%'.format(r2_score(y_test, LReg.predict(X_test))))
            XGB = xgb (X_train, y_train)
            st.write('R-squared value for eXtreme Gradient Boosting Regression predictor on test set: {:.2f}%'.format(r2_score(y_test, XGB.predict(X_test))))
            LassReg = lassoReg (X_train, y_train)
            st.write('R-squared value for Lasso Regression predictor on test set: {:.2f}%'.format(r2_score(y_test, LassReg.predict(X_test))))

        
        
        
        

		
##data = pd.read_csv('./sample_data/california_housing_test.csv')
##data.head()
##
##data.shape[0]
##
##for column in data:
##    if (data[column].nunique() == data.shape[0]):
##        data.drop([column], axis=1, inplace=True)
##
##for column in data:
##    if 'name' in column.lower():
##        data.drop([column], axis=1, inplace=True)
##
##data.info()
##
##data.isnull().any().sum()
##
##obj_df = data.select_dtypes(include=['object']).copy()
##data=data.select_dtypes(exclude=['object'])
##try:
##  one_hot = pd.get_dummies(obj_df)#,drop_first=True)
##except:
##  one_hot = pd.DataFrame()
##
##main_df = pd.concat([one_hot,data],axis=1)
##
##main_df.head()
##
##def normalize (x): 
##    return ( (x-np.min(x))/ (max(x) - min(x)))
##
##main_df = main_df.apply(normalize)
##
##main_df.head()
##
###prompt for Target
###target = 'Exited'
##target = 'median_house_value'
##
##from sklearn.model_selection import train_test_split
##
##X = main_df.drop([target],axis=1)
##y = main_df[target]
##X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100)
##
##from sklearn.ensemble import GradientBoostingClassifier
##gradientBoosting = GradientBoostingClassifier()
##gradientBoosting.fit(X_train, y_train);
##
##print('Accuracy of Gradient Boosting classifier on test set: {:.2f}'.format(gradientBoosting.score(X_test, y_test)))
##
##from sklearn.metrics import classification_report
##
##pred_y=gradientBoosting.predict(X_test)
##print(classification_report(y_test, pred_y))
##
##import xgboost as xgb
##from sklearn.metrics import mean_squared_error
##
##xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
##                max_depth = 5, alpha = 10, n_estimators = 10)
##
##xg_reg.fit(X_train,y_train)
##preds = xg_reg.predict(X_test)
##
##rmse = np.sqrt(mean_squared_error(y_test, preds))
##print("RMSE: %f" % (rmse))

if __name__ == '__main__':
	main()
