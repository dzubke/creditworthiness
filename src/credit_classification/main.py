# standard libraries
import pickle

# 3rd party libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression


# project libraries
from explore_data import describe, unique_values
from prep_data import read_csv, NOT_startswith, remove_prefix, add_column, drop_columns, remove_suffix, int_converter, float_converter, object_columns, count_nan
from prep_data import nan_to_zero
from model import model_fit
from assess_model import F1score, plot_confusion_matrix


def main() -> None:
    """The main method that calls of the function to perform the analysis

    """

    # path to cleaned dataset
    label_csv = '/Users/dustin/CS/jobs/interview_problems/Zume/CreditClassification/data/csv/dataset_cleaned.csv'

    # reading the data into a pandas dataframe
    data_df = read_csv(label_csv)

    describe(data_df)
    print(data_df.columns.values)
    print(unique_values(data_df['loan_status']))
    print(type(data_df["loan_status"]))
    
    # these words need to be removed from the loan_status labels. a new feature will be extracted using them as well
    prefix = 'Does not meet the credit policy. Status:'

    # creates the column of whether a loan passed the credit policy and adds it to the dataframe
    credit_policy = NOT_startswith(prefix, data_df['loan_status'])
    print(credit_policy)
    # data_df = add_column(credit_policy, 'credit_policy', data_df)

    data_df['loan_status'] = remove_prefix(prefix, data_df['loan_status'])
    print(unique_values(data_df['loan_status']))

    # now convert the words in the loan_status column into binary values
    zero_label = 'Charged Off'
    # everything that doesn't start with 'Charged Off' which is 'Fully Paid' is assigned a value of one
    data_df['loan_status'] = NOT_startswith(zero_label, data_df['loan_status'])
    # print(binary_labels.values)

    # adding the new binary labels
    # data_df = add_column(binary_labels, 'labels', data_df)

    # drop the loan_status colummn
    # data_df = drop_columns(data_df, col_list = ['loan_status'])
    # data_df = data_df.drop(columns='loan_status', axis=1)
    # print(data_df)

    # removing the characters ' months' from the column 'term'
    suffix_months = ' months'
    data_df['term'] = remove_suffix(suffix_months, data_df['term'])
    print(data_df.iloc[:, :5])
    describe(data_df)
    
    # the columns of 'term' are still string values so we need to convert them
    print(type(data_df['term'][0]))
    # so convert them to ints
    data_df['term'] = int_converter(data_df['term'])
    # and they are ints. 
    print(type(data_df['term'][0]))
    
    # Who else needs to be ints? Let's look - 
    print(data_df.iloc[:, :10])

    # need to drop an interest rate sign off of int_rate. we can do that with remove_suffix and convert to float with float_converter
    suffix_percent = '%'
    data_df['int_rate'] = remove_suffix(suffix_percent, data_df['int_rate'])
    data_df['int_rate'] = float_converter(data_df['int_rate'])

    # the same will be done with revol_util. These two functions should probably be wrapped in another function
    # data_df['revol_util'] = remove_suffix(suffix_percent, data_df['revol_util'])
    # data_df['revol_util'] = float_converter(data_df['revol_util'])
    # I'm getting a "AttributeError: 'float' object has no attribute 'rstrip'" error when I run remove_suffix on 'revol_util' so I'm going to come back to it

    # and they are floats
    # print(type(data_df['int_rate'][0]))

    # whats next...

    # show me the money...
    # print(data_df.iloc[:, -10:])
    describe(data_df)

    # ok at this point, we are just going to rip out everything that isn't a float, int, or bool and get a basic model running. As per usual, we will come back from more data cleaning
    col_list = object_columns(data_df)

    print('there are 18 columns with object type as shown in the dtypes in the describe function')
    print(f'We can call the object_column function to find there are: {len(col_list)} objects detected')
    print(col_list)


    slim_data_df = drop_columns(data_df, col_list)

    # the two lines below may be garbage... to be disposed of later
    # slim_data_df['mths_since_last_delinq'] = pd.to_numeric(slim_data_df['mths_since_last_delinq'])
    # slim_data_df['mths_since_last_delinq'] = slim_data_df['mths_since_last_delinq'].fillna(0)

    # these columns I think need to be deteled for no good reason... other than they look fishy...
    fishy_drop_list = ['next_pymnt_d', 'desc', 'mths_since_last_delinq', 'mths_since_last_record' ]
    slim_data_df = drop_columns(slim_data_df, fishy_drop_list)

    print(slim_data_df.iloc[:, 10:20])
    describe(slim_data_df)

    # the 'id' column needs to be removed because the values aren't meaningful
    id_drop_list = ['id', ]
    slim_data_df = drop_columns(slim_data_df, id_drop_list)
    describe(slim_data_df)

    nan_list = ['annual_inc', 'delinq_2yrs', 'inq_last_6mths', 'open_acc', 'pub_rec', 'total_acc', 'pub_rec_bankruptcies']
    slim_data_df = nan_to_zero(slim_data_df, nan_list)
    describe(slim_data_df)
    print(f"{slim_data_df.iloc[:, 21:]}")


    mask = count_nan(slim_data_df)
    print(f"count NaN: {mask}")

    # separating out the y-labels
    y_labels = slim_data_df.pop('loan_status')
    y_mask = count_nan(y_labels)
    print(f"y_mask: {y_mask}")

    lr = LogisticRegression()

    lr_fit = model_fit(lr, slim_data_df, y_labels)

    # pickling the model
    with open('model_fit.pickle', 'wb') as model_fn:
        pickle.dump(lr_fit, model_fn)
    
    # creating a sample of the data to test in the server_Helper.py model
    data_excerpt = slim_data_df.iloc[0,:].to_numpy().reshape(1,-1)

    # pickling that data sample to be unpickled in the server_Helper.py model
    with open('data_sample.pickle', 'wb') as data_fn:
        pickle.dump(data_excerpt, data_fn)

    print(f"data excerpt: {data_excerpt}, size: {data_excerpt.shape}")

    print(int(lr_fit.predict(data_excerpt)[0]))

    lr_F1score_train = F1score(lr_fit, slim_data_df, y_labels)
    print(f"F1 score for training set: {lr_F1score_train}")

    print(f"data sampe: {slim_data_df.iloc[0,:]}")
    # lr_F1score_dev = F1score(lr_fit, slim_data_df, y_labels)


    # plot_confusion_matrix(lr_fit, slim_data_df, y_labels)




    #$ save_fn = r'/Users/dustin/CS/jobs/interview_problems/Zume/CreditClassification/data'
    # plot_datatable(slim_data_df, slim_data_df, save_fn)
    # test_print()



    # print(type(data_df['desc'][0]))

    """SCRATCH CODE

    # these columns have dates in them and I want to drop them off for now to get some preliminary results
    # dates_drop_list = ['last_credit_pull_d', 'last_pymnt_d', 'issue_d','earliest_cr_line' ]
    # slim_data_df = drop_columns(slim_data_df, dates_drop_list)

    # misc_drop_list = ['addr_state', 'zip_code', 'title', 'emp_title', ]
    # slim_data_df = drop_columns(slim_data_df, misc_drop_list)

    # non-exclusive list of columns to be made into classifications
    # class_drop_list = ['purpose','verification_status', 'home_ownership', 'grade', 'sub_grade']
    # slim_data_df = drop_columns(slim_data_df, class_drop_list)

    # columns that need to be converted into numeric values
    # numeric_drop_list = ['revol_util', 'emp_length']
    # slim_data_df = d.
    # rop_columns(slim_data_df, numeric_drop_list)

    """




if __name__ == "__main__":  main()