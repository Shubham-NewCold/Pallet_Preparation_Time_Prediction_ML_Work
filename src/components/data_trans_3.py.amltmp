# #We just want the input to the DataTransformationConfig

from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object  # Used for saving the pickle file
import sys
from dataclasses import dataclass
import os

class CompleteTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to derive the target variable.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        # Create a copy to avoid modifying the original data
        df = df.copy()
        try:
            logging.info(f"Starting transformation process...")
            logging.info(f"Shape before preprocessing: {df.shape}")
            
            # Convert datetime columns
            datetime_columns = ['Buffer Assign', 'Planned Arrival', 'latestPick']
            for col in datetime_columns:
                df[col] = pd.to_datetime(df[col], format='%d/%m/%Y %H:%M', errors='coerce')
            logging.info(f"Shape after converting datetime columns: {df.shape}")

            # Handle empty or zero values in 'latestPick' column
            #df['latestPick'] = df['latestPick'].replace(['', '0'], pd.NaT)
            df['latestPick'] = df['latestPick'].fillna(df['Buffer Assign'])
            logging.info(f"Shape after handling 'latestPick': {df.shape}")

            # Feature engineering: Extract datetime features
            df['hour'] = df['Buffer Assign'].dt.hour
            df['month'] = df['Buffer Assign'].dt.month
            df['quarter'] = df['Buffer Assign'].dt.quarter
            logging.info(f"Shape after adding hour, month, and quarter: {df.shape}")

            # Cyclic encoding for Quarter (1 to 4)
            df['Quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
            df['Quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
            logging.info(f"Shape after adding Quarter cyclic encoding: {df.shape}")

            # Day of week and DayType features
            df['weekday'] = df['Buffer Assign'].dt.weekday
            df['DayType'] = df['weekday'].apply(lambda x: 'Weekday' if x in [0, 1, 2, 3, 4] else 'Weekend')
            df['DayType_Weekend'] = df['DayType'].map({'Weekend': 1, 'Weekday': 0})
            logging.info(f"Shape after adding DayType features: {df.shape}")

            # Time-based features
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 4)).astype(int)
            df['hour_of_week'] = df['weekday'] * 24 + df['hour']
            df['pallets_in_buffer'] = df['Confirmed Pallets'] - df['PickedPallets']
            logging.info(f"Shape after adding time-based features: {df.shape}")

            # Further time features
            df['buffer_occupancy_ratio'] = df['pallets_in_buffer'] / df['Confirmed Pallets'].replace(0, 1e-6)
            df['time_since_latestPick'] = ((df['Buffer Assign'] - df['latestPick']).dt.total_seconds() / 60)
            df['time_since_latestPick'] = df['time_since_latestPick'].fillna(0)
            logging.info(f"Shape after adding time_since_latestPick: {df.shape}")

            # More time-based and density features
            df['truck_pallet_time_delay'] = ((df['Planned Arrival'] - df['Buffer Assign']).dt.total_seconds() / 60)
            df['pallet_density'] = df['Confirmed Pallets'] / (df['LoadSequence Count'] + 1)
            df['pick_pallet_ratio'] = df['PickedPallets'] / df['Confirmed Pallets']
            df['order_size'] = df['Confirmed Pallets'] / df['Order Count']
            df['sequence_order_ratio'] = df['LoadSequence Count'] / df['Order Count']
            df['pallets_in_sequence_ratio'] = df['LoadSequence Count'] / df['Confirmed Pallets']
            df['weekend_workload'] = df['DayType_Weekend'] * df['Confirmed Pallets']
            df['trip_size_factor'] = (df['Confirmed Pallets'] * df['Order Count']) / (df['LoadSequence Count'] + 1)
            logging.info(f"Shape after adding additional features: {df.shape}")

            # Frequency encoding for categorical columns
            freq_encoding_carrier = df['Carrier'].value_counts()
            df['Carrier'] = df['Carrier'].map(freq_encoding_carrier).fillna(0)

            freq_encoding_client = df['Client'].value_counts()
            df['Client'] = df['Client'].map(freq_encoding_client).fillna(0)

            carrier_avg_pallets = df.groupby('Carrier')['Confirmed Pallets'].transform('mean')
            df['carrier_avg_pallets'] = carrier_avg_pallets

            mean_arrival_time_by_carrier = df.groupby('Carrier')['Planned Arrival'].transform('mean')
            df['arrival_time_deviation'] = (df['Planned Arrival'] - mean_arrival_time_by_carrier).dt.total_seconds() / 60
            logging.info(f"Shape after encoding categorical features: {df.shape}")

            # Cyclic encoding for Day of Week
            df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
            df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
            logging.info(f"Shape after adding DayOfWeek cyclic encoding: {df.shape}")

            # Cyclic encoding for Hour of Day
            df['Hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['Hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            logging.info(f"Shape after adding Hour cyclic encoding: {df.shape}")

            # Cyclic encoding for Month of Year
            df['Month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            logging.info(f"Shape after adding Month cyclic encoding: {df.shape}")

            # Add recency-based features
            df['time_since_last_buffer_assign'] = ((df['Buffer Assign'] - df['Buffer Assign'].shift(1)).dt.total_seconds() / 60).fillna(0)
            df['recent_pallet_density'] = df['pallet_density'].shift(1).fillna(0)
            df['recent_pick_pallet_ratio'] = df['pick_pallet_ratio'].shift(1).fillna(0)
            logging.info(f"Shape after adding recency features: {df.shape}")

            # Add interaction-based features
            df['hour_of_week_x_pallets_in_buffer'] = df['hour_of_week'] * df['pallets_in_buffer']
            df['hour_of_day_x_pallet_density'] = df['hour'] * df['pallet_density']
            df['Confirmed_Pallets_x_LoadSequence_Count'] = df['Confirmed Pallets'] * df['LoadSequence Count']
            df['Order_Count_x_Pallet_Density'] = df['Order Count'] * df['pallet_density']
            df['Carrier_x_Order_Count'] = df['Carrier'] * df['Order Count']
            df['Client_x_Order_Size'] = df['Client'] * df['order_size']
            logging.info(f"Shape after adding interaction features: {df.shape}")

            # Drop the specified columns after feature engineering
            columns_to_drop = ['Client', 'Carrier', 'Trip.(MLS)', 'Trip Status', 'Buffer Assign', 
                               'ReadyToLoad', 'Prep Time per PLT', 'Planned Arrival', 'latestPick', 
                               'DayType']
            df = df.drop(columns=columns_to_drop, errors='ignore')
            logging.info(f"Shape after dropping specified columns: {df.shape}")

            # Selecting the final features
            selected_features = [
                'Confirmed Pallets', 'time_since_latestPick', 'truck_pallet_time_delay', 
                'arrival_time_deviation', 'time_since_last_buffer_assign', 'recent_pallet_density', 
                'hour_of_week_x_pallets_in_buffer', 'Client_x_Order_Size', 'hour_of_week', 
                'recent_pick_pallet_ratio', 'hour_of_day_x_pallet_density', 'Carrier_x_Order_Count', 
                'pallets_in_buffer', 'carrier_avg_pallets', 'order_size', 'pallet_density', 
                'DayOfWeek_sin'
            ]
            df = df[selected_features]
            logging.info(f"Shape after selecting final features: {df.shape}")
            
            return df
        
        except Exception as e:
            logging.error(f"Error occurred during transformation: {str(e)}")
            raise CustomException(e, sys)
    
    def fit_transform(self, X, y=None):
        """
        Combines fit and transform steps.
        """
        return self.fit(X, y).transform(X)


@dataclass
class DataTransformationConfig3:
    train_data_final_path: str=os.path.join("artifacts","train_final.csv") #all data will be saved in artifacts path, filename train.csv
    #os.path dynamically adjust / or \ based on os
    #output => artifacts\train.csv
    test_data_final_path: str=os.path.join("artifacts","test_final.csv") #all data will be saved in artifacts path
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl") #Create models and want to save in a pkl file

class DataTransformation3:
    def __init__(self):
        self.data_transformation_config3 = DataTransformationConfig3()

    def get_data_transformer_object(self):
        """
        Returns a preprocessor pipeline for deriving the target variable.
        """
        try:
            # Define the pipeline with the custom transformer
            target_variable_pipeline = Pipeline([
                ("complete_transformation_including_target_variable", CompleteTransformer())
            ])
            

            return target_variable_pipeline
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation2(self, train_data_path, test_data_path) -> pd.DataFrame:
        try:
            logging.info(f"Reading path for CSV files for train and test")
            # Step-1: Loading the Train and Test data from the path provided
            train_df = pd.read_csv(train_data_path)

            print("null in train_df", train_df.isnull().sum())
            test_df = pd.read_csv(test_data_path)

            print("Train Shape", train_df.shape)
            print("Test Shape", test_df.shape)

            ##################################################################################################################
            ###### Step-1.5: Preprocessing which is just specific with regards
            # to model building and will not be required in preprocessor.pkl
            # Preprocessing steps to derive the Target Variable [Remember,
            # it is not present in dataset]
            ####################################################################################################################

            # T.1 - Deriving the Target Variable
            # Derive Target Variable for Train DataFrame
            print("Shape Before Transformation - T.1 - Derive Target Variable for Train:", train_df.shape)
            train_df['ReadyToLoad'] = pd.to_datetime(train_df['ReadyToLoad'], format='%d/%m/%Y %H:%M', errors='coerce')
            train_df['Buffer Assign'] = pd.to_datetime(train_df['Buffer Assign'], format='%d/%m/%Y %H:%M', errors='coerce')
            train_df['target_variable'] = ((train_df['ReadyToLoad'] - train_df['Buffer Assign']).dt.total_seconds() / 60) / train_df['Confirmed Pallets']  # Calculate target variable
            print("Shape After Transformation - T.1 - Derive Target Variable for Train:", train_df.shape)

            # Derive Target Variable for Test DataFrame
            print("Shape Before Transformation - T.1 - Derive Target Variable for Test:", test_df.shape)
            test_df['ReadyToLoad'] = pd.to_datetime(test_df['ReadyToLoad'], format='%d/%m/%Y %H:%M', errors='coerce')
            test_df['Buffer Assign'] = pd.to_datetime(test_df['Buffer Assign'], format='%d/%m/%Y %H:%M', errors='coerce')
            test_df['target_variable'] = ((test_df['ReadyToLoad'] - test_df['Buffer Assign']).dt.total_seconds() / 60) / test_df['Confirmed Pallets']  # Calculate target variable
            print("Shape After Transformation - T.1 - Derive Target Variable for Test:", test_df.shape)

            ################################################

            # T.2 Dropping duplicates and null

            # Checking for Null and Duplicated Values
            null_count = train_df['Carrier'].isnull().sum()
            print(f"Number of null values in 'Carrier' column: {null_count}")

            # Count the number of duplicate 'Trip.(MLS)' values
            duplicate_count = train_df.duplicated(subset=['Trip.(MLS)'], keep=False).sum()

            # Print the count of duplicate rows
            print(f"Number of rows with duplicate 'Trip.(MLS)' values: {duplicate_count}")

            # Print the number of rows before removing nulls and duplicates
            initial_row_count = len(train_df)
            print(f"Number of rows before removing nulls and duplicates: {initial_row_count}")

            # Remove rows with null values in the 'Carrier' column
            train_df.dropna(subset=['Carrier'], inplace=True)

            # Remove duplicate rows based on 'Trip.(MLS)' and drop all occurrences of duplicates (keep=False)
            train_df.drop_duplicates(subset=['Trip.(MLS)'], keep=False, inplace=True)

            # Verify the number of rows after removal
            final_row_count = len(train_df)
            print(f"Number of rows after removing nulls and duplicates: {final_row_count}")

            # Drop records where 'Buffer Assign' > 'ReadyToLoad'
            train_df.drop(train_df[train_df['Buffer Assign'] >= train_df['ReadyToLoad']].index, inplace=True)

            # Verify the number of rows after removal
            final_row_count = len(train_df)
            print(f"Number of rows after dropping records where 'Buffer Assign' > 'ReadyToLoad': {final_row_count}")

            # Count the number of rows with 0 or negative values in 'target_variable'
            count_negative_or_zero = (train_df['target_variable'] <= 0).sum()

            # Print the count
            print(f"Number of rows with 0 or negative values in 'target_variable': {count_negative_or_zero}")

            # # Box plot to visualize outliers in the 'target_variable' before treatment
            # plt.figure(figsize=(8, 6))
            # # sns.boxplot(x=train_df['target_variable'])
            # plt.title("Boxplot of Target Variable Before Treatment")
            # plt.show() 

            # Calculate Q1, Q3, IQR, upper and lower limits for the 'target_variable'
            q1 = train_df['target_variable'].quantile(0.10)
            q3 = train_df['target_variable'].quantile(0.90)
            iqr = q3 - q1
            ul = q3 + 1.5 * iqr  # Upper Limit
            ll = q1 - 1.5 * iqr  # Lower Limit

            # Count outliers before clipping
            outliers_count = train_df[(train_df['target_variable'] < ll) | (train_df['target_variable'] > ul)].shape[0]
            print(f"Number of outliers in 'target_variable' before treatment: {outliers_count}")

            # Clip outliers in the 'target_variable'
            train_df['target_variable'] = train_df['target_variable'].clip(lower=ll, upper=ul)

            # Count outliers after clipping
            outliers_count_after = train_df[(train_df['target_variable'] < ll) | (train_df['target_variable'] > ul)].shape[0]
            print(f"Number of outliers in 'target_variable' after treatment: {outliers_count_after}")

            # Box plot to visualize outliers in the 'target_variable' after treatment
            # plt.figure(figsize=(8, 6))
            # sns.boxplot(x=train_df['target_variable'])
            # plt.title("Boxplot of Target Variable After Treatment")
            # plt.show()

            # Step-2: Removing the Target Variable from the dataframe and sending it for preprocessing
            target_column_name = "target_variable"
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            null_count = target_feature_train_df.isnull().sum()

            print("######################Checking Shape before Merge######################")
            print("shape of target_feature_train_df", target_feature_train_df.shape)
            print("shape of input_feature_train_df", input_feature_train_df.shape)
            print("#######################################################################")

            print("nan in input_feature_train_df_final", train_df.isna().sum())

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            print("Information of DataFrame")
            print(train_df.info())

            # Step-2: Sending the DataFrame with Input Features for preprocessing
            preprocessing_obj = self.get_data_transformer_object()

            input_feature_train_df = input_feature_train_df.reset_index(drop=True)
            target_feature_train_df = target_feature_train_df.reset_index(drop=True)

            input_feature_train_df_final = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_df_final = preprocessing_obj.fit_transform(input_feature_test_df)

            # Step-3: Rejoining the train_df_final & test_df_final back with the Target Variable
            # Export the DataFrame to a CSV file
            train_df_final = pd.concat([input_feature_train_df_final, target_feature_train_df], axis=1)
            test_df_final = pd.concat([input_feature_test_df_final, target_feature_test_df], axis=1)

            print("######################Checking Shape After Merge######################")
            print("Shape of input_feature_train_df_final", input_feature_train_df_final.shape)
            print("shape of target_feature_train_df", target_feature_train_df.shape)
            print("Shape of train_df_final", train_df_final.shape)
            print("#######################################################################")

            print("nan in input_feature_train_df_final", train_df_final.isna().sum())

            # Step 4: Save the transformed data to CSV files - this is just a precautionary measure
            logging.info("Saving transformed train and test datasets.")
            train_df_final.to_csv(
                self.data_transformation_config3.train_data_final_path, index=False, header=True
            )

            test_df_final.to_csv(
                self.data_transformation_config3.test_data_final_path, index=False, header=True
            )

            # Step 5: Saving the preprocessing object
            save_object(
                # to save the pickle file
                file_path=self.data_transformation_config3.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Transformation of the datasets into final CSV files is completed.")
            return (
                # Step 4: Returning the path of the final csv file paths for next step
                train_df_final,
                test_df_final,
                self.data_transformation_config3.train_data_final_path,
                self.data_transformation_config3.test_data_final_path,
                self.data_transformation_config3.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e, sys)
