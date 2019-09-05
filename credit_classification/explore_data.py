# standard libraries
from typing import Tuple, List


# non-standard libraries
import pandas as pd

def describe(in_df: pd.DataFrame) -> None:
    """This function takes in a pandas dataframe as input and outputs a variety of information about the dataframe. 

    Parameters
    ----------
    in_df: pd.DataFrame
        a pandas dataframe

    Returns
    ----------
    Not sure yet. May only print stuff...
    
    """

    print(in_df.info())


def plot(in_df: pd.DataFrame) -> None:
    """this fuction will plot the individual columns of the dataset


    """

    pass

def unique_values(in_df: pd.DataFrame) -> List: 
    """This function will take in a pandas data series as input and will return a set of unique values. This will be used to see how many unique values are in the Series.

    Parameters
    -----------
    in_df: pd.DataFrame
        A column of a pandas dataframe (so a data series?)

    Returns
    -------
    List
        A list of all the unique values in the pandas dataframe

    """
    
    return in_df.unique().tolist()