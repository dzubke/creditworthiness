# model.py
# standard libraries
from typing import Callable, List, Any, Dict


# 3rd partylibraries
import pandas as pd
import numpy as np
    # ML models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier



def model_object_dict() -> Dict[str, Any]:
    """Creates a dictionary of model objects to test on the input data

    Parameters
    ----------
    None

    Returns
    ----------
    model_dict: Dict[str, Any]

    """

    # models to test: logistic regression, naive bayes, multi-layer perceptron neural network, k-nearest neighbors, support vector machine classifier, 
    # and a gradient boosted classifier

    logreg = LogisticRegression(solver = 'lbfgs')
    naive_bayes = GaussianNB()
    MLPC = MLPClassifier(hidden_layer_sizes= (50, 100, 50, 25, 10), activation='relu', solver='adam')
    knn = KNeighborsClassifier(n_neighbors= 5)
    svc = SVC()
    gbc = GradientBoostingClassifier()      #xgboost is a faster implementation but in manys is a similar implementation as sklearn gb classifier
    
    model_dict =    {"logreg": logreg, 
                    "naive_bayes": naive_bayes, 
                    "multi_layer_perceptron_NN": MLPC,
                    "knn": knn,
                    "svc": svc,
                    "gbc": gbc
                    }

    return model_dict


def model_fit(model, Xtrain: pd.DataFrame, ytrain: pd.DataFrame):
    """Creates an sklearn LogisticRegression object and fits the Xtrain and ytrain input data to it.
    """

    model_fit = model.fit(Xtrain, ytrain)

    return model_fit


def model_fit_all(model_dict: Dict[str, Any], Xtrain: pd.DataFrame, ytrain: pd.DataFrame) -> Dict[str, Any]:
    """This function takes a Dictionary of model objects as input and returns a fit of that model using the x_data and y_labels

    Parameters
    ---------
    model_dict: Dictionary of model objects
    
    Xtrain: pd.DataFrame
        input training that the models in model_list will be trained on
    
    ytrain: pd.DataFrame
        training lables that the models in model_list will be trained on
    
    Returns
    --------
    model_fit_list: Dictionary of model.fit objects
        a dictionary of the fitted model objects

    """

    model_fit_dict: Dict[str, Any] = {}

    for key, model in model_dict.items():
        model_fit_dict.update({key: model_fit(model, Xtrain, ytrain)})
    
    return model_fit_dict

