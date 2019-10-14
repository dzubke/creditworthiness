# standard libraries
from typing import Callable, Tuple, List, Dict, Any
import time

# non-standard libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
# import chart_studio.plotly.plotly as py
# import chart_studio.plotly.figure_factory as ff


def count_time(*args, print_values: bool = True):
    """Not sure if this will work like this, but idealy this function that take in a function with all of the necessary arguments and would run the function
        inside of it and count the amount of time it toook to run the function

    """
    
    start_time = time.time()


    function_outputs = args # call the function

    stop_time = time.time()
    duration = stop_time - start_time
    
    if print_values: 
        print(f"The function took: {duration} seconds")

    return duration, function_outputs


def roc_assess(model_fit, Xinput: np.ndarray,  ylabel: np.ndarray, print_values: bool = True) -> None:
    """
    Parameters
    ----------
    model_fit: sklearn.model.fit
        the object object type it not entirely accurate because sklearn doesn't have a 'model' members. it has 'linear_model' and others, but the general
        ides is that the fit of a sklearn.model needs to be passed as the model_fit argument. 

    Xinput: a 2d np.ndarray
        the input pixel values

    ylabels: a 1d np.ndarray
        the true labels that the y_hat predictions will be compared to

    print_values: bool = True
        if True, the method will print the values of the confusion matrix. If false, it will not print the values. 

    """

    y_hat = model_fit.predict(Xinput)   # column vextor predicted y-values. shape = # samples x 0 

    # predicted_proba outputs the probability of the prediction being zero or one with shape = # samples x 2
    y_prob = model_fit.predict_proba(Xinput)[:,1]   # we only want the probability that y_hat is equal to one, so we only take the right column. 
    
    fpr, tpr, threshold  = roc_curve(ylabel, y_prob)   # roc_curve (Receiver Operating Characteristic) outputs the false positive rates and true positive rates

    roc_auc = auc(fpr, tpr) # area under the cure (auc) of the roc curve

    if print_values: 
        # all of the print statements below were used to help me understand what roc_curve() and auc() were doing
        # print(f"y_hat sample: {y_hat[:15]}, y_hat shape: {y_hat.shape}")
        # print(f"type: {type(y_prob)}, y_score shape: {y_prob.shape}, y_score: {y_prob[:15]}")
        # print(f"fpr type: {type(fpr)}, fpr shape: {fpr.shape}, fpr sample: {fpr}")
        # print(f"tpr type: {type(tpr)}, tpr shape: {tpr.shape}, tpr sample: {tpr}")
        # print(f"thres type: {type(threshold)}, shape: {threshold.shape}, sample: {threshold}")

        print(f"area under the roc cure: {roc_auc}")
    
        plt.figure()
        
        # Plotting our Baseline..
        plt.plot([0,1],[0,1])
        plt.plot(fpr,tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')


def sklearn_score(model_fit, Xtrain, ytrain):

    return model_fit.score(Xtrain, ytrain)

def F1score(model_fit, Xinput: np.ndarray,  ylabel: np.ndarray, print_values: bool = False) -> Tuple[float, float, float]:
    """
    Parameters
    ----------
    model_fit: sklearn.model.fit
        the object object type it not entirely accurate because sklearn doesn't have a 'model' members. it has 'linear_model' and others, but the general
        ides is that the fit of a sklearn.model needs to be passed as the model_fit argument. 

    Xinput: a 2d np.ndarray
        the input pixel values

    ylabels: a 1d np.ndarray
        the true labels that the y_hat predictions will be compared to

    print_values: bool = True
        if True, the method will print the values of the confusion matrix. If false, it will not print the values. 

    """

    y_hat = model_fit.predict(Xinput)   # column vextor predicted y-values. shape = # samples x 0 

    tn, fp, fn, tp = confusion_matrix(ylabel, y_hat).ravel()  # calculates the confusion matrix, which is the unravelled matrix [[true neg, false pos], [false neg, true pos]]
    precision = tp/ (tp + fp)
    recall = tp/ (tp + fn)
    F1_score = 2* precision * recall / (precision + recall)

    if print_values: 
        print (f"F1 score: {F1_score}")
        print (f"Precision: {precision}")
        print (f"Recall: {recall}")

    return F1_score, precision, recall

def F1score_all(model_fits: Dict[str, Any], Xinput: np.ndarray,  ylabel: np.ndarray, print_values: bool = False) -> Dict[str, Tuple[float, float, float]]:
    """this fuction runs the F1score function on all of the model_fit objects in the model_fit dictionary

    Parameters
    ---------
    model_fits: Dict[str, Any]
        a dictionary of fitted sklearn models
    
    Xinput: np.ndarray
        the input data

    ylabels:  np.ndarray
        the true labels that the y_hat predictions will be compared to

    print_values: bool = True
        if True, the method will print the values of the confusion matrix. If false, it will not print the values. 

    Returns
    --------
    F1score_dict = Dict[str, Tuple[float, float, float]]
        a dictionary of whose keys are the model names in model_fits and the values are a Tuple of the output from F1score,
            which are the F1score, precision, and recall of the models

    """

    F1score_dict = {}

    for key, model_fit in model_fits.items():
        F1score_dict.update({key: F1score(model_fit, Xinput, ylabel, print_values = print_values)})

    return F1score_dict


def plot_confusion_matrix(model_fit, Xinput: np.ndarray,  y_label: np.ndarray) -> None:
    
    y_hat = model_fit.predict(Xinput)   # column vextor predicted y-values. shape = # samples x 0
    labels = [0, 1]
    cm = confusion_matrix(y_label, y_hat, labels)
    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


    
    '''
    conf_mat = confusion_matrix(y_label, y_hat, labels=[0,1]) # the confusion matrix without labels
    plt.imshow(conf_mat,  interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    '''


def plot_Table(metric_names: List[str], data_names: List[str], metric_data: List[float], save_bool: bool = False ):
    """This function plots a table metric_data with the column names coming fromm metric_names and the row names coming from data_names

    Parameters
    ---------
    metric_names: List of strings
        the names of the metrics which will be the names of each column

    data_names: List of strings
        the names of the input data, which will be the row names

    data_values: List of floats
        the values to be displayed
    
    save_bool: bool
        a boolean that determines if the the picture of the table is saved to memory



    Returns


    """
    pass

'''
def plot_datatable(input_data: pd.DataFrame, label_data: pd.DataFrame, out_fn: str, save_bool: bool = False ):
    """The function plots the input_data and label_data in a plotly table with the name table_name. 
        If save_bool is set to true, the function saves the picture to disk

    

    """

    #df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminderDataFiveYear.csv')
    # *** create the dataframe ***
    df_sample = input_data[100:120]

    table = ff.create_table(df_sample)
    py.iplot(table, filename=out_fn)
'''