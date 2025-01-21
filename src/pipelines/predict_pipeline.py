
# import sys
# import pandas as pd
# from src.exception import CustomException
# from src.utils import load_object, test_on_unseen_data # to load our pickle file
# from datetime import datetime,date,time
# from src.components.model_trainer import ModelTrainer

# #First Class -> has the init function without nothing, 
# class PredictPipeline:
#     def __init__(self):
#         pass

#     #will simply do prediction
#     # two pckle files we have currently, preprocessor and  model
#     #Prediction - 1: Gives prediction of 1 input
#     def predict(self, file_path, processed_file_path):
#         try:
#             print("########Step-4- Extension - Inside predict_pipeline.py")
#             model_path = 'artifacts/model.pkl'
#             preprocessor_path = 'artifacts/preprocessor.pkl'
#             print("########Step-4- Data Transformation Triggered through preprocessor.pkl")

#             #load_obect we will craete, will load the pickle file
#             model = load_object(file_path = model_path) #should be created in utils 
#             preprocessor = load_object(file_path = preprocessor_path)
#             input_data = pd.read_csv(file_path)
#             # Save the original data for reference
#             original_data = input_data.copy()
#             data_final_to_pred = preprocessor.transform(input_data)

#             print("########Step-4 - End of Data Transformation#########################")
#             print("########Step-5 - Printing the Model Object")
#             print("########Step-6 - Printing Final DataFrame for Prediction")
#             print(data_final_to_pred.head(2))
#             print("########Step-7 - Printing DataFrame Info")
#             print(data_final_to_pred.info())
            
#             # Get predictions (regression output)
#             predictions = model.predict(data_final_to_pred)

#             # Append the predictions to the original data
#             input_data['Predictions'] = predictions

#             # Concatenate the original data with predictions
#             final_df = pd.concat([original_data, input_data[['Predictions']]], axis=1)

#             # Save the final results to the processed file path
#             final_df.to_csv(processed_file_path, index=False)

#         except Exception as e:
#             raise CustomException(e, sys)

import sys
import pandas as pd
from src.exception import CustomException  # Ensure CustomException is defined in src.exception
from src.utils import load_object  # Import load_object to load models
from datetime import datetime, date, time
# from src.components.model_trainer import ModelTrainer  # Ensure ModelTrainer is properly defined if needed

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, file_path, processed_file_path):
        try:
            print("########Step-4- Extension - Inside predict_pipeline.py")
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            print("########Step-4- Data Transformation Triggered through preprocessor.pkl")

            # Load model and preprocessor objects
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Load input data
            input_data = pd.read_csv(file_path)
            original_data = input_data.copy()

            # Preprocess the data
            data_final_to_pred = preprocessor.transform(input_data)

            print("########Step-4 - End of Data Transformation#########################")
            print("########Step-5 - Printing the Model Object")
            print("########Step-6 - Printing Final DataFrame for Prediction")
            print(data_final_to_pred.head(2))
            print("########Step-7 - Printing DataFrame Info")
            print(data_final_to_pred.info())

            # Get predictions (regression output)
            predictions = model.predict(data_final_to_pred)

            # Append predictions to the original data
            input_data['Predictions'] = predictions

            # Concatenate the original data with predictions
            final_df = pd.concat([original_data, input_data[['Predictions']]], axis=1)

            # Save the final results to the processed file path
            final_df.to_csv(processed_file_path, index=False)

        except Exception as e:
            raise CustomException(e, sys)  # Make sure CustomException is correctly defined in src.exception
