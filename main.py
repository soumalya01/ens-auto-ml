#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 6/25/20 8:49 PM 2020

@author: Anirban Das
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report


def main():
    st.title("ABC Corp..")
    st.title("Automated Machine Learning Web (POC)")
    data_file = './DataDump/file' + datetime.now().strftime("%d%b%Y_%H%M%S%f") + '.csv'
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
    X_train = ''
    X_test = ''
    y_train = ''
    y_test = ''
    y_pred = ''

    @st.cache
    def load_data():
        data = pd.read_csv(data_file)
        # st.write(data.head())
        return data

    if st.checkbox("Show Data HEAD or TAIL"):
        select_option = st.radio("Select option", ['HEAD', 'TAIL'])
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

    if st.checkbox("*Select Target Column"):
        all_columns = load_data().columns
        target = st.selectbox("Select", all_columns)
        if dataDF[target].dtype == "object":
            label_encoder = LabelEncoder()
            dataDF[target] = label_encoder.fit_transform(dataDF[target])
        # st.write(load_data()[names])

    if st.checkbox("*Auto Discard Columns"):
        for column in dataDF:
            if dataDF[column].nunique() == dataDF.shape[0]:
                dataDF.drop([column], axis=1, inplace=True)
        for column in dataDF:
            if 'name' in column.lower():
                dataDF.drop([column], axis=1, inplace=True)

        st.text("Data Columns")
        st.write(dataDF.columns)
        st.text("Count of NaN values")
        st.write(dataDF.isnull().any().sum())

    if st.checkbox("*Preprocess Object Type Columns"):
        obj_df = dataDF.select_dtypes(include=['object']).copy()
        dataDF = dataDF.select_dtypes(exclude=['object'])
        try:
            one_hot = pd.get_dummies(obj_df)  # ,drop_first=True)
        except Exception as e:
            print("There has been an exception: ", e)
            one_hot = pd.DataFrame()

        dataDF = pd.concat([one_hot, dataDF], axis=1)

    sc = StandardScaler()
    st.header("Split DataSet into Train and Test")

    if st.checkbox("*Split"):
        print(dataDF.dtypes)
        X = dataDF.drop([target], axis=1)
        # X = X.apply(normalize)
        y = dataDF[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    if st.checkbox("*Normalize Columns"):
        from sklearn.preprocessing import MinMaxScaler
        norm = MinMaxScaler()
        X_train = norm.fit_transform(X_train)
        X_test = norm.transform(X_test)
        X = norm.transform(X)

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

    def gradBoost(X, y):
        from sklearn.ensemble import GradientBoostingClassifier
        gradientBoosting = GradientBoostingClassifier()
        gradientBoosting.fit(X, y)
        return gradientBoosting

    def randForest(X, y):
        from sklearn.ensemble import RandomForestClassifier
        randomForest = RandomForestClassifier()
        randomForest.fit(X, y)
        return randomForest

    def svm(X, y):
        from sklearn import svm
        clf = svm.SVC()
        clf.fit(X, y)
        return clf

    def xgb(X, y):
        import xgboost as xgboost
        xg_reg = xgboost.XGBRegressor()
        xg_reg.fit(X, y)
        return xg_reg

    def linearReg(X, y):
        from sklearn.linear_model import LinearRegression
        lineReg = LinearRegression()
        lineReg.fit(X, y)
        return lineReg

    def lassoReg(X, y):
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=0.01)
        lasso.fit(X, y)
        return lasso



    if st.checkbox("*ML Algorithms"):
        st.write("Available algorithms are:")
        st.write("Binary Classification: GB Classifier, RF Classifier, SVM")
        st.write("Regression: OLS, XGB, Lasso Regression")
        if dataDF[target].nunique() == 2:
            st.header("Using Binary Classification Algorithms")
            GB = gradBoost(X_train, y_train)
            st.write('Accuracy of Gradient Boosting classifier on test set: {:.2f}'.format(GB.score(X_test, y_test)))
            RF = randForest(X_train, y_train)
            st.write('Accuracy of Random Forest classifier on test set: {:.2f}'.format(RF.score(X_test, y_test)))
            SVM = svm(X_train, y_train)
            st.write('Accuracy of SVM classifier on test set: {:.2f}'.format(SVM.score(X_test, y_test)))
        elif dataDF[target].nunique() / dataDF[target].count() < .1:
            st.header("Using Multi-Class Classification Algorithms")
            GB = gradBoost(X_train, y_train)
            st.write('Accuracy of Gradient Boosting classifier on test set: {:.2f}'.format(GB.score(X_test, y_test)))
            st.write(classification_report(y_test, GB.predict(X_test)))
            RF = randForest(X_train, y_train)
            st.write('Accuracy of Random Forest classifier on test set: {:.2f}'.format(RF.score(X_test, y_test)))
            st.write(classification_report(y_test, RF.predict(X_test)))
        else:
            st.header("Using Regression Algorithms")
            from sklearn.metrics import mean_squared_error, r2_score
            LReg = linearReg(X_train, y_train)
            st.write('R-squared value for Linear Regression predictor on test set: {:.2f}%'.format(
                r2_score(y_test, LReg.predict(X_test))))
            XGB = xgb(X_train, y_train)
            st.write('R-squared value for eXtreme Gradient Boosting Regression predictor on test set: {:.2f}%'.format(
                r2_score(y_test, XGB.predict(X_test))))
            LassReg = lassoReg(X_train, y_train)
            st.write('R-squared value for Lasso Regression predictor on test set: {:.2f}%'.format(
                r2_score(y_test, LassReg.predict(X_test))))

    st.header("Run Prediction on Test Set")        
    if st.checkbox("*Select Desired Algorithm"):
        if dataDF[target].nunique() == 2:
            selectML = st.selectbox("Select", ['Gradient Boosting classifier','Random Forest classifier','SVM classifier'])
            if selectML == 'Gradient Boosting classifier':
                dML = GB
            elif selectML == 'Random Forest classifier':
                dML = RF
            elif selectML == 'SVM classifier':
                dML = SVM
        elif dataDF[target].nunique() / dataDF[target].count() < .1:
            selectML = st.selectbox("Select", ['Gradient Boosting classifier','Random Forest classifier'])
            if selectML == 'Gradient Boosting classifier':
                dML = GB
            elif selectML == 'Random Forest classifier':
                dML = RF
        else:
            selectML = st.selectbox("Select", ['Linear Regression predictor','eXtreme Gradient Boosting Regression predictor','Lasso Regression predictor'])
            if selectML == 'Linear Regression predictor':
                dML = LReg
            elif selectML == 'eXtreme Gradient Boosting Regression predictor':
                dML = XGB
            elif selectML == 'Lasso Regression predictor':
                dML = LReg

        data_test = './DataDump/file' + datetime.now().strftime("%d%b%Y_%H%M%S%f") + '.csv'
        file_test = st.file_uploader("Upload test file")
        try:
            if file_bytes is not None:
                with open(data_test, mode='w', newline='') as f:
                    print(file_test.getvalue().strip('\r\n'), file=f)
                    data_load_state.text("Upload....Done!")
                dataDF1 = pd.read_csv(data_test)
        except FileNotFoundError:
            st.error('File not found.')



    if st.checkbox("*PREDICT"):



        for column in dataDF1:
                if dataDF1[column].nunique() == dataDF1.shape[0]:
                        dataDF1.drop([column], axis=1, inplace=True)
        for column in dataDF1:
                if 'name' in column.lower():
                        dataDF1.drop([column], axis=1, inplace=True)

        obj_df1 = dataDF1.select_dtypes(include=['object']).copy()
        dataDF1 = dataDF1.select_dtypes(exclude=['object'])
        try:
            one_hot1 = pd.get_dummies(obj_df1)  # ,drop_first=True)
        except Exception as e:
            print("There has been an exception: ", e)
            one_hot1 = pd.DataFrame()

        dataDF1 = pd.concat([one_hot1, dataDF1], axis=1)

        X1 = dataDF1.drop([target], axis=1)
        y1 = dataDF1[target]
        X1 = norm.transform(X1)

        st.write('Accuracy of Selected Algorithm on test Dataset: {:.2f}'.format(dML.score(X1, y1)))

        



if __name__ == '__main__':
    main()
