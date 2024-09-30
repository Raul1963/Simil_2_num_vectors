import os.path
import time
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import openpyxl
import matplotlib.ticker as mtick
from scipy.spatial.distance import euclidean,cityblock



# def checkSameElems(l1: list, l2: list):
#     f = 0
#     for i in range(len(l1)):
#         if l1[i]==l2[i]:
#             f+=1
#     return f/len(l1)
#
#
# def checkSum(l1: list, l2: list):
#     return min(sum(l1), sum(l2))/max(sum(l1), sum(l2))
#
#
# def checkOrder(l1: list, l2: list):
#     arr1=l1.copy()
#     arr2=l2.copy()
#     start1 = time.perf_counter()
#     arr1.sort()
#     elapsed1 = time.perf_counter()-start1
#     start2 = time.perf_counter()
#     arr2.sort()
#     elapsed2 = time.perf_counter()-start2
#     return elapsed1/elapsed2 if elapsed1 < elapsed2 else elapsed2/elapsed1
#
#
# def compVect(l1: list, l2: list):
#     if l1 == l2:
#         return 1
#     c1 = checkSameElems(l1, l2)
#     c2 = checkSum(l1, l2)
#     c3 = checkOrder(l1, l2)
#     return (c1+c2+c3)/3
#
# def compVector2(l1,l2):
#     result=0
#     for i in range(len(l1)):
#         if l1[i]!=l2[i] and (l1[i]!=0 or l2[i]!=0):
#             result=result + abs(l1[i]-l2[i])/max(l1[i],l2[i])
#     return result/100




# ref1=[0.00028918,0,0.0161943,0.00028918,0,0.0381724,0,0.00028918,0,0.00173511,0,0,0,0,0,0,0,0,0.0037594,0,0.00318103,0.0147484,0.00028918,0,0,0.00260266,0,0,0.0153268,0.00115674,0,0.00173511,0,0,0,0.00144592,0,0,0,0.00404858,0,0,0.00057837,0.00028918,0,0,0.0164835,0,0,0,0,0,0,0,0,0]
# ref2=[0.00151149,0,0.0190447,0.0003023,0,0.0371826,0,0.00181378,0,0.00816203,0.0003023,0,0,0,0,0,0,0,0.00483676,0,0.00060459,0.0175333,0.00090689,0,0,0.00302297,0,0,0.011185,0.00060459,0,0.00060459,0,0,0,0.00211608,0,0,0,0.00423216,0,0,0,0.0003023,0,0,0.0169287,0.00060459,0,0,0,0,0,0.00060459,0,0]
# ref3=[0,0,0.0170743,0.00046147,0,0.0396862,0,0.00092293,0,0.00507614,0.00092293,0,0,0,0,0,0,0,0.00323027,0,0.0013844,0.020766,0.00046147,0,0,0.00230734,0,0,0.0170743,0.0013844,0,0.0013844,0,0,0,0.0013844,0,0,0,0.00415321,0,0,0.00046147,0,0,0,0.0152284,0.00046147,0,0,0,0,0,0.00046147,0,0]
#
# l1=[0,0,0.0142544,0,0,0.0175439,0,0,0,0.0131579,0,0,0,0,0,0,0,0,0.00438596,0,0,0.00657895,0.00219298,0,0,0.00109649,0,0,0.0263158,0,0,0.00109649,0,0,0,0.00219298,0.00109649,0,0,0.00109649,0,0,0,0.00109649,0,0,0.0208333,0.00109649,0,0,0,0,0,0,0,0]
# l2=[0,0,0.00745898,0,0,0.0263551,0,0,0,0.00447539,0,0,0,0,0,0,0,0,0.0014918,0,0,0.0149179,0.00049727,0,0,0.00447539,0,0,0.0134262,0,0,0,0,0,0,0,0,0,0.0014918,0.00099453,0,0,0.00049727,0,0,0,0.0139234,0,0,0,0,0,0,0,0,0]
# l3=[0.00080939,0,0.0149737,0,0,0.0441117,0,0.00121408,0,0.0129502,0,0,0,0,0,0,0,0,0.00242817,0,0.00647511,0.0105221,0,0,0,0.00161878,0,0,0.00809389,0.00202347,0,0.00121408,0,0,0,0.00283286,0,0,0,0,0,0,0,0,0,0,0.0315662,0,0,0,0,0,0,0,0,0]
# l4=[0.00126743,0,0.0139417,0.00126743,0,0.0291508,0,0,0,0.00760456,0,0,0,0,0,0,0,0,0.00126743,0,0.00126743,0.0152091,0,0,0,0.00380228,0,0,0.00760456,0.00126743,0,0.00380228,0,0,0,0,0,0,0,0.00126743,0,0,0.00126743,0,0,0,0.0164765,0,0,0,0,0,0,0.00253485,0,0]
# l6=[0,0,0.00705882,0,0,0.0352941,0,0,0,0.0164706,0,0,0,0,0,0,0,0,0,0,0.00235294,0.0164706,0,0,0,0.00470588,0,0,0.0141176,0.0117647,0,0.00235294,0,0,0,0.00470588,0,0,0,0,0,0,0.00470588,0,0,0,0.0305882,0,0,0,0,0,0,0,0,0]
# l7=[0,0,0.00943396,0,0,0.0314465,0,0.00314465,0,0.00314465,0,0,0,0,0,0,0,0,0.00157233,0,0.00314465,0.00943396,0,0,0,0.00157233,0,0,0.00628931,0,0,0.00314465,0,0,0,0.00157233,0,0,0,0,0,0,0,0,0,0,0.0172956,0,0,0,0,0,0,0.00157233,0,0]
# l8=[0,0,0.012775,0,0,0.0411639,0,0,0,0.00851668,0,0,0,0.00070972,0,0,0,0,0.00141945,0,0,0.0227111,0.00212917,0,0,0.00709723,0,0,0.0156139,0,0,0.00283889,0,0,0,0.00212917,0.00070972,0,0,0.00496806,0,0,0.00283889,0,0.00070972,0,0.0262598,0.00070972,0,0,0,0,0,0.00070972,0,0]
# l9=[0.0003488,0,0.0149983,0,0,0.0338333,0,0.0003488,0,0.00662714,0,0,0,0,0.00209278,0,0,0,0.00209278,0,0.00104639,0.00976631,0,0,0,0.0118591,0,0,0.0101151,0.00244158,0,0.00104639,0,0,0,0.00313917,0,0,0,0.00279037,0,0,0.00069759,0.0003488,0,0.0003488,0.0143007,0,0,0,0,0,0,0.00069759,0,0]
# l=[l1,l2,l3,l4,l6,l7,l8,l9]


#print(euclidean_similarity_percentage(ref2,ref3))



def testl(train,test):
    """
    Function to find the best limit for the conf matrix.
    Requires the dataset name as parameter.
    """
    for i in np.arange(0.013, 0.015, 0.0001):
        #Numpy is used so it can iterate through floats, the first parameter is start value, the second final value and the third the step for each iteration.
        print(f" for {i}:")
        confMatrix(train,test,i)



def calcDist(data):
    """
    Calculates for each element of data the corresponding euclidean distance to each element of the reference array
    """
    refs = data[data['authorCode'] == 5].iloc[:, :-1]
    #Refs will contain only the rows of the file with the target author, in this case 5, but will not contain the last column with the author code.
    distances = []
    for i in range(len(data)):
        comp = []
        for j in range(len(refs)):
            comp.append(cityblock(data.iloc[ i, :-1],refs.iloc[j]))
        if(data['authorCode'].iloc[i]==5):
            res = sum(comp) / (len(comp)-1)
        else:
            res = sum(comp) /len(comp)
        distances.append(res)
    return distances

def trains(traindata):
    """
    Finds the best threshold, for distingushing the results, based on the training data, plots the results and returns the limit.
    """
    newL=0
    bestAcc=0
    distances = calcDist(traindata)
    x = []
    y = []
    trueLabel = [1 if traindata.iloc[i][ 'authorCode'] == 5 else 0 for i in range(len(traindata))]

    for i in np.arange( 0.0, 0.1, 0.0001):
        limit=i
        x.append(i)
        predict = [1 if distance < limit else 0 for distance in distances]
        cm = confusion_matrix( y_true=trueLabel,y_pred=predict, labels=[0,1])
        correct_predictions =  cm[0, 0] + cm[1, 1]
        total_predictions = cm.sum()
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        y.append(int(f"{accuracy*100:.0f}"))
        if(accuracy>bestAcc):
            bestAcc, newL=accuracy, i

    plt.plot(x,y)
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy (%)')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    plt.show()
    return newL

def confMatrix(train,test):
    """
    This function outputs the confusion matrix and the accuracy of the desired dataset.
    It calculates the probability that a row from the dataset belongs to a specific author code.
    It accepts as parameters the datasets name and a limit, which functions like a threshold so the results can be filtered.
    """
    try:
        traindata = pd.read_csv(train)
        testdata = pd.read_csv(test)
    except:
        traindata = pd.read_excel(train)
        testdata = pd.read_excel(test)
    #The file contents are put in an array using the pandas library,along with the corresponding column names. Each element(row) will be an array.

    trueLabels=[1 if testdata.iloc[i]['authorCode']==5 else 0 for i in range(len(testdata))]

    limit=trains(traindata)

    distances=calcDist(testdata)
    #For each array from data it calculates the euclidean distance with every array from refs and appends the result in a new array(comp). Then it sums the elements from comp and divides it by the length of the array.
    #The new value will be the average euclidean distance between one array from data and all arrays from refs and then it appends it to the distances array.

    #This is the limit which will be used to classify the results, to be either 1(true) or 0(false).
    print(f"Best limit: {limit}\n")

    #This is the array, which contains the intended results, 1 for true, 0 for false.
    predict= [1 if distance < limit else 0 for distance in distances]
    #This is the array which contains the actual results, which are filtered by the limit,1 for true, 0 for false.

    cm=confusion_matrix(y_true=trueLabels,y_pred=predict,labels=[0,1])
    #Here the confusion matrix is declared, and is given its true array and its predicted array, alongside what labels(values) appear in the two arrays.

    print("Confusion Matrix:\n",cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=["Not Similar","Similar"])
    disp.plot()
    plt.show()
    #Here the confusion matrix will appear as an image with the row and cloumn headers similar and not similar, in order to find the true, false positives etc.

    correct_predictions = cm[0, 0] + cm[1, 1]
    total_predictions = cm.sum()
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    print(f"Accuracy: {accuracy:.4f}")
    #Here we output the accuracy of our function, by adding the true positives with the true negatives(coorect predictions) and by dividing them by all the elements form the confusion matrix(total predictions)

confMatrix("ROST-P/Custom_train01.csv","ROST-P/custom_test01.csv")