# model.py
from typing import Callable

# non-standard libraries
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np



def model_fit(model, Xtrain: pd.DataFrame, ytrain: pd.DataFrame):
    """Creates an sklearn LogisticRegression object and fits the Xtrain and ytrain input data to it.
    """

    model_fit = model.fit(Xtrain, ytrain)

    return model_fit





def run_model(model: Callable, xtrain: np.ndarray, xtest: np.ndarray, ytrain: np.ndarray, ytest: np.ndarray):
    """

    Parameters
    ----------


    Returns
    -------


    """

    model_fit = model.fit(xtrain, ytrain)

    train_acc=model_fit.score(xtrain, ytrain)
    test_acc=model_fit.score(xtest,ytest)
    print("Training Data Accuracy: %0.2f" %(train_acc))
    print("Test Data Accuracy:     %0.2f" %(test_acc))

    # return LR_Fit

    # lr1 = test_model(lr, xtrain, ytrain, xtest, ytest)



