# standard libraries 
import unittest    
import sys 

# non-standard libaries
import pandas as pd

# project libraries
from src.credit_classification.prep_data import read_csv, remove_prefix, NOT_startswith


sys.path.insert(1, '/Users/dustin/CS/jobs/interview_problems/Zume/CreditClassification/credit_classification')

class TestExploreData(unittest.TestCase):


    def test_read_csv(self):
        '''Tests the read_csv to ensure it is outputting the correct type

        '''

        data_path =r'/Users/dustin/CS/jobs/interview_problems/Zume/CreditClassification/tests/testdata/dataset_cleaned.csv'

        data_df = read_csv(data_path)

        self.assertIsInstance(data_df, pd.DataFrame)


    def test_remove_prefix(self):
        '''Tests the remove prefix method

        '''
        series_1 = pd.Series(['Fully Paid', 'Charged Off', 'Does not meet the credit policy. Status:Fully Paid', 'Does not meet the credit policy. Status:Charged Off'])

        prefix = 'Does not meet the credit policy. Status:'

        self.assertEqual(remove_prefix(prefix, series_1).tolist(), ['Fully Paid', 'Charged Off', 'Fully Paid', 'Charged Off'])


    def test_NOT_startswith(self):
        '''Tests the NOT_startswith method

        '''
        series_1 = pd.Series(['Fully Paid', 'Charged Off', 'Does not meet the credit policy. Status:Fully Paid', 'Does not meet the credit policy. Status:Charged Off'])

        prefix = 'Does not meet the credit policy. Status:'

        self.assertEqual(NOT_startswith(prefix, series_1).tolist(), [True, True, False, False])
