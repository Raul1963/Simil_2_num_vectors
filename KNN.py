import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def traintest(train,test):
    """
    Reads and scales and transforms the train and test data, to be more efficient.
    """
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
    y_test = y_test.apply( lambda x: 1 if x == 5 else 0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled,X_test_scaled,y_train,y_test

def testK(train,test):
    """
    Finds the optimal k(k-th neighbour) so that it has the best accuracy possible, plots the results.
    """
    X_train_scaled,X_test_scaled,y_train,y_test=traintest(train,test)
    x=[]
    y=[]
    for i in range(1,30):
        x.append(i)
        knn = KNeighborsClassifier(n_neighbors=i,p=1)
        knn.fit(X_train_scaled, y_train)

        y_pred = knn.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        y.append(int(f"{accuracy*100:.0f}"))
        print(f" for {i} Accuracy: {accuracy:.4f}")
    plt.plot(x,y)
    plt.xlabel("Nearest neighbour")
    plt.ylabel("Accuracy (%)")
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter( decimals=0))
    plt.show()

# testK()
def knn(train,test):
    """
    Uses the knn classifier with the train and test data to try to predict the desired results and also it calculates the accuracy of the method and prints the confusion matrix.
    """
    X_train_scaled,X_test_scaled,y_train,y_test=traintest(train,test)
    knn = KNeighborsClassifier(n_neighbors=4,p=1)
    knn.fit(X_train_scaled, y_train)

    y_pred = knn.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    cm = confusion_matrix(y_test, y_pred, labels=np.unique([y_test,y_pred]))
    print("Confusion Matrix:\n", cm)


    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique([y_test,y_pred]))
    disp.plot()
    plt.show()

knn("ROST-P/ROST-P-trainSet1.csv","ROST-P/ROST-P-testSet1.csv")
testK("ROST-P/ROST-P-trainSet1.csv","ROST-P/ROST-P-testSet1.csv")