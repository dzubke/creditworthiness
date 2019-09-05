# standard libraries
from typing import List

# non-standard libraries
import pandas as pd


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

    data = pd.read_csv(data_path, sep=",", header=0)

    return data




def clean(df):
    """


    """

    # pd.dropna     #drops empty cells
    pass


def one_hot(in_df: pd.DataFrame) -> pd.DataFrame:
    """This is one hot function (ha) that converts a input dataframe and converts into a one-hot matrix.

    """

    pass