import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score 
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay

def main():
    ################ Step 1 Create Web Title #####################
    st.title("Binary Classification Streamlit App")
    st.sidebar.title("Binary Classification Streamlit App")
    st.markdown(" เห็ดนี้กินได้หรือไม่??? 🍄‍🟫🍄‍🟫🍄‍🟫")
    st.sidebar.markdown(" เห็ดนี้กินได้หรือไม่??? 🍄‍🟫🍄‍🟫🍄‍🟫")

    ############### Step 2 Load dataset and Preprocessing data ##########
    
    @st.cache_data(persist=True)
    def load_data():
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        file_path = os.path.join(DATA_DIR, 'mushrooms.csv')

        data = pd.read_csv(file_path)
        return data

    def handle_missing_values(df):
        # จัดการ Missing Values โดยแทนค่าที่หายไปด้วยค่าที่เกิดบ่อยที่สุด
        for col in df.columns:
            if df[col].dtype == 'object':  # ตรวจสอบว่าเป็น Categorical
                mode_value = df[col].mode()[0]  # ค่าที่เกิดบ่อยที่สุด
                df[col] = df[col].replace('?', mode_value)  # แทนค่าที่หายไป
        return df

    @st.cache_data(persist=True)
    def preprocess_data(df):
        df = handle_missing_values(df)
        label = LabelEncoder()
        for col in df.columns:
            df[col] = label.fit_transform(df[col])
        return df

    @st.cache_data(persist=True)
    def spliting_data(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, ax=ax, display_labels=class_names)
            st.pyplot(fig)
        
        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            fig, ax = plt.subplots()
            RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            fig, ax = plt.subplots()
            PrecisionRecallDisplay.from_estimator(model, x_test, y_test, ax=ax)
            st.pyplot(fig)

    # Start main section here.
    df = load_data()
    df = preprocess_data(df)
    x_train, x_test, y_train, y_test = spliting_data(df)
    class_names = ['edible', 'poisonous']
    st.sidebar.subheader("Choose Classifiers")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vectore Machine (SVM)", "Logistic Regression", "Random Forest"))


     ############### Step 3 Train a SVM Classifier ##########

    if classifier == 'Support Vectore Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma  = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Supper Vector Machine (SVM) results")
            model = SVC(random_state=None, C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test, y_test)
            y_pred   = model.predict(x_test)

            precision = precision_score(y_test, y_pred).round(2)
            recall = recall_score(y_test, y_pred).round(2)
            
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", precision)
            st.write("Recall: ", recall)
            plot_metrics(metrics)


    

     ############### Step 4 Training a Logistic Regression Classifier ##########
     # Start you Code here #
    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        solver = st.sidebar.selectbox("Solver", ("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"), key='solver')
        C = st.sidebar.number_input("C", 0.01 , 2.0, step=0.01, key='C')
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistice Regression results")
            # model = SVC(C=C, kernel=kernel, gamma=gamma)
            model = LogisticRegression(random_state=None ,solver=solver, max_iter=500, C=C)
            model.fit(x_train, y_train)
            # model.fit(x_train,y_train)
            accuracy = model.score(x_test, y_test)
            # accuracy = model.score(x_test, y_test)
            y_predict = model.predict(x_test)
            # y_pred   = model.predict(x_test)
            
            precision = precision_score(y_test, y_predict).round(2)
            # precision = precision_score(y_test, y_pred).round(2)
            recall = recall_score(y_test, y_predict).round(2)
            # recall = recall_score(y_test, y_pred).round(2)
            
            st.write("Accuracy: ", round(accuracy, 2))
            # st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", precision)
            # st.write("Precision: ", precision)
            st.write("Recall: ", recall)
            # st.write("Recall: ", recall)
            plot_metrics(metrics)
            # plot_metrics(metrics)


     ############### Step 5 Training a Random Forest Classifier ##########
    # Start you Code here #
    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("จำนวนของ decision trees (ยิ่งเยอะยิ่งแม่นยำ)",
                                         100, 150, step=5,
                                         key="n_estimators"
                                        )
        max_depth = st.sidebar.number_input("การตั้งค่าความลึกจำกัดจะช่วยลด overfitting",
                                        5, 30, step=5,
                                        key="max_depth"   
                                     )
        min_samples_split = st.sidebar.number_input("จำนวนตัวอย่างขั้นต่ำที่ต้องมีในแต่ละ node",
                                                2, 20, step=1,
                                                key="min_samples_split"
                                             )
        min_samples_leaf = st.sidebar.number_input("min_samples_leaf สูงจะช่วยลด overfitting",
                                                1, 2, step=1,
                                                key="min_samples_leaf"
                                            )
        
        metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

        if st.sidebar.button("Classify", key="classify"):
            st.subheader("Random Forest results")
            model = RandomForestClassifier(
                                            random_state=None, 
                                            n_estimators=n_estimators,
                                            max_depth=max_depth,
                                            min_samples_split=min_samples_split,
                                            min_samples_leaf=min_samples_leaf,
                                        )
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_predict = model.predict(x_test)
            
            precision = precision_score(y_test, y_predict).round(2)
            recall = recall_score(y_test, y_predict).round(2)
            
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", precision)
            st.write("Recall: ", recall)
            plot_metrics(metrics)


    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom dataset")
        st.write(df)


if __name__ == '__main__':
    main()


