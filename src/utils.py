import os
import sys
import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from sklearn.utils.validation import _deprecate_positional_args



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def test_on_unseen_data (model, unseen):
    """
    This function takes a regression model and unseen data as input,
    then returns the predicted continuous values for the unseen data.
    """
    try:
        # Use the model to make predictions on the unseen data
        y_pred = model.predict(unseen)
        return y_pred  # Return the predicted continuous values

    except Exception as e:
        return f"Error during prediction: {e}"
