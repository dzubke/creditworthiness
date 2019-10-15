# standard libraries
from typing import List, Tuple, Any, Dict
import re
import pickle


# 3rd party libraries
import pandas as pd
import numpy as np
from numpy import array, argmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder



def read_csv(data_path: str) -> any:
    """This function reads in data formattaed as a csv file.

    Parameters
    ----------
    data_path: str
        the path of the dataset in the csv format

    Returns
    ---------
    Not sure yet

    """

    assert type(data_path) == str, "input path is not a string"

    data = pd.read_csv(data_path, sep=",", header=0, index_col = 'id')

    return data


def NOT_startswith(prefix: str, in_series: pd.Series) -> pd.Series:
    """
    This function returns a pd.Series of boolean values if the elements of a pd.Series do not start with the inputted prefix. In other words, if an element of the series starts with the inputted prefix, a value of False
        will be recorded in the output series.

    Parameters
    ----------
    prefix: str
        the input string that will return a False value if the series element starts with that string
    
    loan_status: pd.Series
        the input labels of the dataset.

    Returns
    --------
    pass_credit_policy: pd.Series
        a column of the data that binary classifies whether a loan met the credit policy (True) or did not meet the credit policy (False)

    """
    assert type(prefix) == str, "the input prefix is not a string"
    assert type(in_series) == pd.Series, "the input array is not a pd.Series"

    out_series = ~in_series.str.startswith(prefix)

    return out_series


def remove_prefix(prefix: str, in_series: pd.Series) -> pd.Series :
    """
    This function removes the input prefix from the input series

    Parameters
    ----------
    prefix: str
        the start of the string that will be removed from all elements in the Series
    
    in_series: pd.Series
        The original labeled data in the dataset

    Returns
    ----------
    out_series: pd.Series
        the loan_status column with the prefix "Does not meet the credit policy. Status:" removed so only "Fully Paid" and "Charged Off" remain

    """

    assert type(in_series) == pd.Series, "input object was not of pd.Series type"

    out_series = in_series.map(lambda x: x.lstrip(prefix)) 

    return out_series


def remove_suffix(suffix: str, in_series: pd.Series) -> pd.Series :
    """
    This function removes the input prefix from the input series

    Parameters
    ----------
    prefix: str
        the start of the string that will be removed from all elements in the Series
    
    in_series: pd.Series
        The original labeled data in the dataset

    Returns
    ----------
    out_series: pd.Series
        the loan_status column with the prefix "Does not meet the credit policy. Status:" removed so only "Fully Paid" and "Charged Off" remain

    """

    # assert type(in_series) == pd.Series, "input object was not of pd.Series type"
    #assert type(in_series[0]) == str, "the elements of the pd.Series must be string type"


    out_series = in_series.map(lambda x: x.rstrip(suffix)) 

    return out_series


def add_column(add_series: pd.Series, col_name: str, in_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function adds the Series add_series to the in_df and returns the updated dataframe. add_series will have the label col_nume in the update dataframe.

    Parameters
    ---------
    add_series: pd.Series
        the series to be added to in_df
    
    col_name: str
        the name of the add_series column when added to in_df

    in_df: pd.DataFrame
        the input dataframe that add_series will be added to

    Returns
    -------
    in_df: pd.DataFrame
        the output dataframe with add_series 

    """

    assert type(add_series) == pd.Series, "add_series is not a pd.Series"
    assert type(col_name) == str, "col_name is not a str"
    assert type(in_df) == pd.DataFrame, "in_df is not a pd.DataFrame"

    # the col_name variable doesn't get assiend in the .assign() call... need to fix this otherwise all the added values will have the anem col_name
    return in_df.assign( col_name = add_series.values)


def drop_columns(in_df: pd.DataFrame, col_list: List[str]) -> pd.DataFrame: 
    """
    This function drops the column with name col_name from the input dataframe in_df

    Parameters
    ---------
    in_df: pd.DataFrame
        input dataframe

    col_name: str
        name of the column to be dropped

    Returns
    ----------
    in_df: pd.DataFrame

    """
    
    for col_name in col_list:
        in_df = in_df.drop(columns=col_name)

    return in_df


def int_converter(in_series: pd.Series) -> pd.Series :
    """
    This function conversts the values in a pd.Series into integers

    Parameters
    ----------
    in_series: pd.Series
        the in_series that will be converted to integers

    Returns
    ----------
    in_series: pd.Series
        the same series with the columns as int values

    """

    assert type(in_series) == pd.Series, "input object was not of pd.Series type"

    return in_series.map(lambda x: int(x)) 


def float_converter(in_series: pd.Series) -> pd.Series :
    """
    This function conversts the values in a pd.Series into floats

    Parameters
    ----------
    in_series: pd.Series
        the in_series that will be converted to floats

    Returns
    ----------
    in_series: pd.Series
        the same series with the columns as float values

    """

    assert type(in_series) == pd.Series, "input object was not of pd.Series type"

    return in_series.map(lambda x: float(x)) 


def object_columns(in_df: pd.DataFrame) -> List['str'] :
    """
    This function returns a list of the column names that have type 'object' (or do not have an int, float, bool data type)

    Parameters
    ----------
    in_df: pd.DataFrame

    Returns
    ----------
    col_list: List['str']
        a list of the column names with type object
    """

    assert type(in_df) == pd.DataFrame, "input object was not of pd.DataFrane type"

    col_list = []
    for label, content in in_df.iteritems():
        if type(content.iloc[0]) == str:
            col_list.append(label)

    return col_list


def count_nan(in_df: pd.DataFrame) -> pd.DataFrame:
    """counts the number of NaN values in each column of the dataframe"""

    mask = in_df.isna().sum()

    return mask


def nan_to_zero(in_df: pd.DataFrame, col_list: List[str]) -> pd.DataFrame:
    """converts NaN values to zeros""

    ---------
    in_df: pd.DataFrame
        input dataframe

    col_name: str
        name of the column to be dropped

    Returns
    ----------
    in_df: pd.DataFrame

    """
    assert type(in_df) == pd.DataFrame, "input object was not of pd.DataFrane type"

    
    for col_name in col_list:
        in_df[col_name] = in_df[col_name].fillna(0)

    return in_df


def dataset_split(data_array: np.ndarray, label_array: np.ndarray, split_ratios: Dict[str, float]
            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """This function splits the datasets into training, development, and test sets based on the float percentages given
        in split_ratios

    Parameters
    ----------
    data_array: np.ndarray of shape (# of images x 19200 pixels)
        The input array of flattened images

    label_array: np.ndarray of shape (# of images x 1 )
        The label array of whether a image contains a ship (1) or not (0)

    split_ratios: dictionary of floats with the keys "train", "dev", and "test" identifying each float ratio
        Dictionary of three float values that determine the percentages of how the training, development, and test sets are determined.
        The first value sets the percentage of the dataset in the training set, second float for the dev set, and third for the test set.
    
    Returns
    --------
    X_train: np.ndarray of shape (# of images * split_ratios[0] x 19200 )
        The training dataset 

    X_dev: np.ndarray of shape (# of images * split_ratios[1] x 19200 )
        The development dataset 

    X_test: np.ndarray of shape (# of images * split_ratios[2] x 19200 )
        The test dataset 
    
    y_train: np.ndarray of shape (# of images * split_ratios[0] x 1 )
        The training labels
    
    y_dev: np.ndarray of shape (# of images * split_ratios[1] x 1 )
        The labels for the development set
    
    y_test: np.ndarray of shape (# of images * split_ratios[2] x 1 )
        The labels of the test set

    """

    assert sum(x for x in split_ratios.values()) <= 1 and sum(x for x in split_ratios.values()) > 0.99, "The sum of the split ratios are not approximately 1"


    # sklearn.train_test_split only splits the data into two groups. This will be called twice but to ensure the dataset
    # ratios are correct we need to perform the computation below to determine the split ratios of both calls of the splitting function
    # As a example, say the dataset is a 0.8/0.1/0.1 split. The first call of train_test_split splits the data into 0.9 train_temp and 0.1 dev.
    # The second call of train_test splits the 0.9 train_temp into a (0.8/0.9) split resulting in a 0.8 training set and a 0.1 test set

    split_ratio_1 = 1.0 - split_ratios.get("dev")       
    split_ratio_2 = split_ratios.get("train")/ split_ratio_1 

    #splits the data into a temporary training set and the original dev set. 
    X_train_temp, X_dev, y_train_temp, y_dev = train_test_split(data_array, label_array, train_size = split_ratio_1, random_state = 1234, stratify = label_array, shuffle = True)
    
    # takes the temporary training sets as input and splits it into the original train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X_train_temp, y_train_temp, train_size = split_ratio_2, random_state = 1234, stratify = y_train_temp, shuffle = True)

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def pickle_object(in_object: Any, out_path: str) -> None:
    """this funciton will pickle the in_object out the filepath out_path

    Parameters
    ----------
    in_object: Any data type
        the object that will be pickled

    out_path: str
        the path where the pickle of the in_object will be saved

    Returns
    -------
    None
        Nothing is returned. the pickled object is saved at out_path
    """

    with open(out_path, 'wb') as data_fn:
        pickle.dump(in_object, data_fn)


def percent_convert(in_series: pd.Series) -> pd.Series:
    """Converts string percentages into floats as decimals versions of the percentages"""
    pass

def one_hot_encoders(in_df: pd.DataFrame, col_list:List[str]) -> pd.DataFrame:
    """This function calls the one_hot_encoder function on all of the column names in col_list

    Parameters
    ---------- 
    in_df: pd.DataFrame
        the input dataframe to be manipulated
    col_name: str
        the name of the column to be converted into a one-hot representation

    Returns
    --------
    in_df: pd.DataFrame
        a modified version of the in_df
    """
    # loop over all of the column names in col_list
    for col_name in col_list:
        in_df = one_hot_encoder(in_df, col_name)
    
    return in_df


def one_hot_encoder(in_df: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """This function converts the column with col_name in dataframe in_df into a one-hot representation of the unique values
        in col_name. It also deletes the col_name column from the in_df. The new one-hot columns will have names of the format:
        'col_name' + '-' + 'value'
    
    Parameters
    ---------- 
    in_df: pd.DataFrame
        the input dataframe to be manipulated
    col_name: str
        the name of the column to be converted into a one-hot representation

    Returns
    --------
    in_df: pd.DataFrame
        a modified version of the in_df
    """
    
    # create a column vector to pass into the OneHotEncoder
    col_values = in_df[col_name].to_numpy().reshape(-1,1)

    # create the OneHotEncoder object
    onehot_encoder = OneHotEncoder(sparse=False)
    
    # fit the object to the col_values
    onehot_encoded = onehot_encoder.fit_transform(col_values)

    # make a list of the encoder column names
    encoder_columns = [col_name+'-'+category for category in onehot_encoder.categories_[0]]
    
    # convert to pd.DataFrame 
    ohe_df = pd.DataFrame(data = onehot_encoded, index = in_df.index, columns=encoder_columns )

    # join in_df and the one hot encoder dataframe ohe_df
    in_df = in_df.join(ohe_df, lsuffix='_caller', rsuffix='_ohe_df')

    # drop the original column
    in_df = drop_columns(in_df, [col_name])

    return in_df




if __name__ == "__main__":
    """
    Scratch code
    ------------

    test_df = pd.Series(['Fully Paid', 'Charged Off', 'Does not meet the credit policy. Status:Fully Paid', 'Does not meet the credit policy. Status:Charged Off'])
    print(test_df)
    # new_test_df = parse_labels(test_df)
    # print(new_test_df)

    prefix = "Does not meet the credit policy. Status:"
    len_prefix = len(prefix)
    print(len_prefix)
    regex = re.compile(prefix)

    for index, rows in test_df.iteritems(): 
        print(rows)
        if regex.match(rows):  
            test_df.iloc[index] = test_df.iloc[index][len_prefix:]
            print(test_df.iloc[index])

    print(test_df)
    """
