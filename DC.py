import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def findParam(X_train,y_train,dtree):
    # The parameters which will be tested with each other.
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 5, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10, 15, 20, 30],
        'min_samples_leaf': [1, 2, 4, 6, 8,10],
        'max_features': [None,  'sqrt', 'log2', 0.25, 0.5, 0.75],
        'max_leaf_nodes': [None,  10, 20, 30, 40, 50, 100],
        'min_impurity_decrease': [0.0,  0.01,  0.05, 0.1, 0.2]
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=dtree,param_grid=param_grid, cv=5,n_jobs=-1,verbose=1, scoring='accuracy')
    grid_search.fit( X_train, y_train)

    # Output the best parameters
    print(grid_search.best_params_)
    return grid_search.best_estimator_, grid_search.cv_results_


def plot_accuracy_evolution(cv_results):
    mean_test_scores = cv_results['mean_test_score']
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mean_test_scores)), mean_test_scores, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Mean Accuracy')
    plt.title('Evolution of Accuracy during Grid Search')
    plt.grid(True)
    plt.show()

def decision_tree(train, test):
    try:
        traindata = pd.read_csv(train)
        testdata = pd.read_csv(test)
    except:
        traindata = pd.read_excel(train)
        testdata = pd.read_excel(test)

    # Extract features and labels
    X_train = traindata.iloc[:, :-1]
    y_train = traindata['authorCode']

    X_test = testdata.iloc[:, :-1]
    y_test = testdata['authorCode']

    # Create and train the Decision Tree classifier
    # dtree = DecisionTreeClassifier(criterion= 'entropy', max_depth= None, max_features= 0.5, max_leaf_nodes= None, min_impurity_decrease= 0.0, min_samples_leaf= 2, min_samples_split= 2, splitter= 'random',random_state=42)
    dtree = DecisionTreeClassifier(random_state=42)
    dtree,cv_results=findParam(X_train,y_train,dtree)
    plot_accuracy_evolution( cv_results)
    dtree.fit(X_train, y_train)

    # Make predictions
    y_pred = dtree.predict(X_test)

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


decision_tree("ROST-P/ROST-P-trainSet1.csv","ROST-P/ROST-P-testSet1.csv")
