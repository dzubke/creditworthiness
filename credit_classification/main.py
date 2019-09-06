from explore_data import describe, unique_values
from prep_data import read_csv, parse_labels

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
    clean_loan_status, credit_policy = parse_labels(data_df['loan_status'])
    print(unique_values(clean_loan_status))
    # describe()






if __name__ == "__main__":
    main()