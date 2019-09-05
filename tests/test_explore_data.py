# standard libraries 
import unittest    
import sys 

# non-standard libaries
import pandas as pd

# project libraries
from credit_classification.prep_data import read_csv     


sys.path.insert(1, '/Users/dustin/CS/jobs/interview_problems/Zume/MLProblem/credit_classification')

class TestExploreData(unittest.TestCase):


    def test_read_csv(self):
        '''Tests the read_csv to ensure it is outputting the correct type

        '''

        data_path =r'/Users/dustin/CS/jobs/interview_problems/Zume/MLProblem/tests/testdata/dataset_cleaned.csv'

        data_df = read_csv(data_path)

        self.assertIsInstance(data_df, pd.DataFrame)