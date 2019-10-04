# system modules
from typing import List

# third-party modules
import numpy as np





def server_model(input_exapmles: List[np.ndarray]) -> bool: 
    """This function will be called by the swagger.yml file to run the prediction model on the input_examples

    Parameters
    ----------
    input_examples: List[np.ndarray]
        A list of loan applications that will be predicted
    



    """