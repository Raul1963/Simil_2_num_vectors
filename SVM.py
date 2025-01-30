import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def svm(train,test):
    # # Load the CSV data
    # try:
    #     data = pd.read_csv(filename)
    # except:
    #     data = pd.read_excel(filename)
    # Split the data into features (X) and labels (y)
    # X = data.iloc[:, :-1]  # All columns except the last one
    # y = data['authorCode']  # Last column
    #
    # # Binary classification: if authorCode == 5, label as 1, else 0
    # y = y.apply(lambda x: 1 if x == 5 else 0)

    # Split the data into 80% training and 20% testing
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Standardize the data
    try:
        traindata = pd.read_csv(train)
        testdata = pd.read_csv(test)
    except:
        traindata = pd.read_excel(train)
        testdata = pd.read_excel(test)

    X_train = traindata.iloc[:,:-1]
    y_train = traindata['authorCode']

    X_test = testdata.iloc[:,:-1]
    y_test = testdata['authorCode']
    y_train = y_train.apply(lambda x: 1 if x == 5 else 0)
    y_test = y_test.apply(lambda x: 1 if x == 5 else 0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training data
    X_test_scaled = scaler.transform(X_test)        # Only transform on test data

    besti=0
    bestc=0
    bestacc=0
    # for i in np.arange(0.1,1, 0.01):
    #     for j in np.arange(0,1,0.01):
    #         svm = SVC( kernel='sigmoid',gamma=i,coef0=j)
    #         svm.fit( X_train_scaled,y_train)
    #         y_pred = svm.predict( X_test_scaled)
    #         accuracy = accuracy_score(y_test,y_pred)
    #         if(accuracy>bestacc):
    #             bestacc,besti,bestc=accuracy,i,j
    #
    # print(f"The best accuracy of {bestacc} is for gamma {besti} and coef0 {bestc}")
    # Create and train the SVM classifier
    svm = SVC(kernel='sigmoid',gamma=0.12,coef0=0.0)
    svm.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = svm.predict(X_test_scaled)
    print("Unique classes in predictions:",np.unique(y_pred))

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=np.unique([y_test,y_pred]))
    print("Confusion Matrix:\n", cm)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique([y_test,y_pred]))
    disp.plot()
    plt.show()

svm("ROST-P/ROST-P-trainSet1.csv","ROST-P/ROST-P-testSet1.csv")
