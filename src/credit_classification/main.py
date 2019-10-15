# standard libraries


# 3rd party libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier



# project libraries
from explore_data import describe, unique_values
from prep_data import read_csv, NOT_startswith, remove_prefix, add_column, drop_columns, remove_suffix, int_converter, float_converter, object_columns, count_nan
from prep_data import nan_to_zero, dataset_split, pickle_object, one_hot_encoders
from model import model_fit, model_object_dict, model_fit_all
from assess_model import F1score, plot_confusion_matrix, F1score_all


def main() -> None:
    """The main method that calls of the function to perform the analysis

    """

    # path to cleaned dataset
    label_csv = '/Users/dustin/CS/jobs/interview_problems/Zume/CreditClassification/data/csv/dataset_cleaned.csv'

    # reading the data into a pandas dataframe
    data_df = read_csv(label_csv)
    print(data_df.index)

    describe(data_df)
    print(data_df.columns.values)
    print(unique_values(data_df['loan_status']))
    print(type(data_df["loan_status"]))
    print(data_df.iloc[0:10, 0:15])
    

    # REMOVING PREFIXES AND SUFFIXES

    # removing prefixes
    prefix = 'Does not meet the credit policy. Status:'     # removing this prefix from the loan_status column

    data_df['loan_status'] = remove_prefix(prefix, data_df['loan_status'])
    print(unique_values(data_df['loan_status']))

    # removing suffixes
    suffix_months = ' months'       # removing this suffix from the term column
    data_df['term'] = remove_suffix(suffix_months, data_df['term'])

    suffix_percent = '%'        # removing the string '%' percent sign from the int_rate and revol_util column
    data_df['int_rate'] = remove_suffix(suffix_percent, data_df['int_rate'])
    data_df['revol_util'] = remove_suffix(suffix_percent, data_df['revol_util'])


    # DATA TYPE CONVERSIONS
    
    # binary conversions
    # Converting the labels 'Charged Off' and 'Fully Paid' in the loan_status column into binary values
    zero_label = 'Charged Off'
    # everything that start with 'Charged Off'  is assigned a zero-value, 
    data_df['loan_status'] = NOT_startswith(zero_label, data_df['loan_status'])
 
    # string to int conversions
    # the columns of 'term' are converted from strings to integers
    data_df['term'] = int_converter(data_df['term'])
    
    # string to float
    # convert  int_rate and revol_util to float
    data_df['int_rate'] = float_converter(data_df['int_rate'])
    data_df['revol_util'] = float_converter(data_df['revol_util'])

    #one-hot conversions
    one_hot_columns = ['verification_status', 'home_ownership', 'grade', 'sub_grade']
    data_df =  one_hot_encoders(data_df, one_hot_columns)
    describe(data_df)
    print(data_df.iloc[0:10, 0:15])


    # ok at this point, we are just going to rip out everything that isn't a float, int, or bool and get a basic model running. As per usual, we will come back from more data cleaning
    col_list = object_columns(data_df)

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
    '''
    split_ratios = {"train": 0.8, "dev": 0.1, "test": 0.1}
    X_train, X_dev, X_test, y_train, y_dev, y_test = dataset_split(slim_data_df, y_labels, split_ratios)

    print(f"X_train shape: {X_train.shape}, X_dev shape: {X_dev.shape}, X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}, y_dev shape: {y_dev.shape}, y_test shape: {y_test.shape}")

    model_dict = model_object_dict()

    model_fits = model_fit_all(model_dict, X_train, y_train)

    F1scores = F1score_all(model_fits, X_dev, y_dev)
    print(F1scores)
    '''




    '''
    logreg = LogisticRegression(solver = 'lbfgs')
    naive_bayes = GaussianNB()
    MLPC = MLPClassifier(hidden_layer_sizes= (50, 100, 50, 25, 10), activation='relu', solver='adam')
    knn = KNeighborsClassifier(n_neighbors= 5)
    svc = SVC()
    gbc = GradientBoostingClassifier()




    lr_fit = model_fit(logreg, X_train, y_train)
    nb_fit = model_fit(naive_bayes, X_train, y_train)
    mlpc_fit = model_fit(MLPC, X_train, y_train)
    knn_fit = model_fit(knn, X_train, y_train)
    svc_fit = model_fit(svc, X_train, y_train)
    gbc_fit = model_fit(gbc, X_train, y_train)


    # pickling the model
    # pickle_object(lr_fit, 'model_fit.pickle')


    # lr_F1score_train = F1score(lr_fit, X_train, y_train, print_values=False)
    # print(f"lr F1 score for training set: {lr_F1score_train}")

    lr_F1score_dev= F1score(lr_fit, X_dev, y_dev, print_values=False)
    print(f"lr F1 score for dev set: {lr_F1score_dev}")

    # lr_F1score_test= F1score(lr_fit, X_test, y_test, print_values=False)
    # print(f"lr F1 score for test set: {lr_F1score_test}")

    nb_F1score_dev= F1score(nb_fit, X_dev, y_dev, print_values=False)
    print(f"nb F1 score for dev set: {nb_F1score_dev}")

    mlpc_F1score_dev = F1score(mlpc_fit, X_dev, y_dev, print_values=False)
    print(f"mlpc F1 score for dev set: {mlpc_F1score_dev}")

    knn_F1score_dev = F1score(knn_fit, X_dev, y_dev, print_values=False)
    print(f"knn F1 score for dev set: {knn_F1score_dev}")
    
    svc_F1score_dev = F1score(svc_fit, X_dev, y_dev, print_values=False)
    print(f"svc F1 score for dev set: {svc_F1score_dev}")

    gbc_F1score_dev = F1score(gbc_fit, X_dev, y_dev, print_values=False)
    print(f"gbc F1 score for dev set: {gbc_F1score_dev}")
    '''

    

    # lr_F1score_dev = F1score(lr_fit, slim_data_df, y_labels)


    # plot_confusion_matrix(lr_fit, slim_data_df, y_labels)
    




    
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

    #$ save_fn = r'/Users/dustin/CS/jobs/interview_problems/Zume/CreditClassification/data'
    # plot_datatable(slim_data_df, slim_data_df, save_fn)
    # test_print()
    # print(type(data_df['desc'][0]))

    # print(f"data sampe: {slim_data_df.iloc[0,:]}")

    # creating a sample of the data to test in the server_Helper.py model
    data_excerpt = slim_data_df.iloc[0,:].to_numpy().reshape(1,-1)

    # pickle the data_excerpt object for testing in server_Helper.py
    pickle_object(data_excerpt, 'data_sample.pickle')
    print(f"data excerpt: {data_excerpt}, size: {data_excerpt.shape}")
    print(int(lr_fit.predict(data_excerpt)[0]))


    """




if __name__ == "__main__":  main()