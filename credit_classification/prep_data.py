# standard libraries
from typing import List, Tuple
import re

# non-standard libraries
import pandas as pd
from numpy import array, argmax
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder


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


def parse_labels(loan_status: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """This function removes the prefix "Does not meet the credit policy. Status:" from the column "loan_status" and creates a separate column with 
    the binary classification "Does not meet the credit policy".

    Parameters
    ----------
    loan_status: pd.Series
        The original labeled data in the dataset

    Returns
    ----------
    cleaned_loan_status: pd.Series
        the loan_status column with the prefix "Does not meet the credit policy. Status:" removed so only "Fully Paid" and "Charged Off" remain
    
    credit_policy: pd.Series
        a column of the data that binary classifies whether a loan met the credit policy (1) or did not meet the credit policy (0)

    Notes
    ------
    The reasoning for this method is that the unique labels in the dataset in the "loan_status" column have the values:
    ['Fully Paid', 'Charged Off', 'Does not meet the credit policy. Status:Fully Paid', 'Does not meet the credit policy. Status:Charged Off']
    As can be seen there is a "Does not meet the credit polcicy prefix in two of the labels". A google search revealed a spare set of explanation, but the 
    one I found here: https://forum.lendacademy.com/?topic=2427.msg20813#msg20813 explains what I suspected: that lending club basically combined two pieces
    of information into one column for some of the loans. For those with the "Does not meet.." prefix, the lender didn't meet the credit policy but were
    still offered a loan and that loan was either Fully Paid or Charged off. This presents an opportunity to use the "Does not meet.." prefix into another 
    feature.

    """

    # this code won't work with a DataFrame due to indexing in the for loop, it must be a Series
    assert type(loan_status) == pd.Series, "input object was not of pd.Series type"

    # creating the regex objects to check which rows have the prefix
    prefix = "Does not meet the credit policy. Status:"
    len_prefix = len(prefix)
    regex = re.compile(prefix)
    
    # the pandas series that will contain the binary values if the example meets the credit policy
    credit_policy = pd.Series()

    for index, rows in loan_status.iteritems(): 
        if regex.match(rows):  
            loan_status.iloc[index] = loan_status.iloc[index][len_prefix:]
            credit_policy.append(pd.Series([0]))
        else:
            credit_policy.append(pd.Series([1]))
    
    return loan_status, credit_policy

    # data['result'] = data['result'].map(lambda x: x.lstrip('+-').rstrip('aAbBcC'))


def clean(df):
    """


    """

    # pd.dropna     #drops empty cells
    pass


def one_hot_encoder(in_df: pd.DataFrame) -> pd.DataFrame:
    """This is one hot function (ha) that converts a input dataframe and converts into a one-hot matrix.

    Parameters
    ----------


    Returns
    --------

    Notes
    -------
    Code for the one-hot encoder taken from: https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/

    """
    # one_hot = np.eye(D)[V.reshape(-1)].T

        
    # define example
    data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
    values = array(data)
    print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)
    # invert first example
    inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    print(inverted)
    


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
