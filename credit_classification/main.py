from explore_data import describe, unique_values
from prep_data import read_csv, NOT_startswith, remove_prefix, add_column, drop_columns, remove_suffix, int_converter, float_converter, object_columns
import pandas as pd

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
    data_df = add_column(credit_policy, 'credit_policy', data_df)

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
    print(data_df)

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

    # need to drop an interest rate sign off of int_rate. we can do that with remove_suffix and convert to int with int_converter

    suffix_percent = '%'
    data_df['int_rate'] = remove_suffix(suffix_percent, data_df['int_rate'])
    data_df['int_rate'] = float_converter(data_df['int_rate'])

     # and they are floats
    print(type(data_df['int_rate'][0]))

    # whats next...

    # show me the money...
    print(data_df.iloc[:, -10:])
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


    # this columns I think need to be deteled for not good reason... other than they look fishy...
    fishy_drop_list = ['next_pymnt_d', 'desc', 'mths_since_last_delinq', 'mths_since_last_record' ]
    slim_data_df = drop_columns(data_df, fishy_drop_list)

    print(slim_data_df.iloc[:, -10:])
    describe(slim_data_df)


    # print(type(data_df['desc'][0]))



    






if __name__ == "__main__":
    main()