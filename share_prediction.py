import argparse
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, r2_score


class SharePredictionModel:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """
        Load CSV data from the specified file path and convert the 'timeslot_datetime_from'
        column to timestamp format.

        Returns:
            pandas.DataFrame: DataFrame containing the loaded data.
        """
        logging.info("Loading data from file: %s", self.file_path)
        data = pd.read_csv(self.file_path, low_memory=False)

        # Convert timeslot_datetime_from from object to timestamp format
        data['timeslot_datetime_from'] = pd.to_datetime(data['timeslot_datetime_from'])

        data = data.sort_values(by=['channel_id', 'main_ident', 'timeslot_datetime_from'])
        logging.info("Data dimensions after loading: %s", data.shape)
        return data

    def process_data(self, data, duplicate_rows=False):
        """
        Perform data processing on the DataFrame.

        Parameters:
            data (DataFrame): DataFrame containing the CSV data.
            duplicate_rows (bool): Flag to indicate whether to duplicate rows if duplicated.
                                    If True, duplicates rows; if False, does not duplicate rows.

        Returns:
            DataFrame: Processed DataFrame.
        """
        if duplicate_rows:
            # Drop duplicated rows
            data.drop_duplicates(inplace=True)
            logging.info("Data dimensions after dropping duplicated rows: %s", data.shape)

        else:
            # Keep duplicated rows
            logging.info("Data dimensions when skipping duplicating: %s", data.shape)

        # Drop columns with just NULL values
        data.dropna(axis=1, how='all', inplace=True)
        logging.info("Data dimensions after dropping columns with just NULL values: %s", data.shape)

        # Drop rows containing NULL values
        data.dropna(axis=0, how='any', inplace=True)
        logging.info("Data dimensions after dropping rows containing NULL values: %s", data.shape)

        return data

    def select_features(self, data, include_main_ident=False, include_share_15_54_3mo_mean=True):
        """
        Select features to include or exclude from the DataFrame.

        Parameters:
            data (DataFrame): DataFrame containing the CSV data.
            include_main_ident (bool): Flag to include or exclude 'main_ident' column. Default is True.
            include_share_15_54_3mo_mean (bool): Flag to include or exclude 'share_15_54_3mo_mean' column. Default is True.

        Returns:
            DataFrame: DataFrame with selected features.
        """

        # Select all features by default
        all_features = list(data.columns)

        # Exclude features based on flags (using list comprehension)
        if not include_main_ident:
            all_features = [col for col in all_features if col != 'main_ident']
        if not include_share_15_54_3mo_mean:
            all_features = [col for col in all_features if col != 'share_15_54_3mo_mean']

        # Return the original DataFrame if no exclusions were made
        return data if all_features == list(data.columns) else data[all_features]

    def extract_datetime_features(self, data, enable=False):
        """
        Extract datetime features from the timeslot_datetime_from column.

        Parameters:
            data (DataFrame): DataFrame containing the CSV data with 'timeslot_datetime_from' column.
            enable (bool): Flag to enable/disable datetime feature extraction. Default is True.

        Returns:
            DataFrame: DataFrame with extracted datetime features.
        """
        if not enable:
            logging.info("Datetime feature extraction is disabled. Skipping...")
            return data

        else:
            logging.info("Generating new features: day_of_week, month_of_year, hour_of_day and season")

            data['day_of_week'] = data['timeslot_datetime_from'].dt.day_name()
            data['month_of_year'] = data['timeslot_datetime_from'].dt.month_name()
            data['hour_of_day'] = data['timeslot_datetime_from'].dt.hour

            # Define seasons based on months
            month_to_season = {
                1: 'Winter', 2: 'Winter', 3: 'Spring',
                4: 'Spring', 5: 'Spring', 6: 'Summer',
                7: 'Summer', 8: 'Summer', 9: 'Fall',
                10: 'Fall', 11: 'Fall', 12: 'Winter'
            }
            data['season'] = data['timeslot_datetime_from'].dt.month.map(month_to_season)

            return data

    def encode_categorical_features(self, data):
        """
        Encode categorical features using one-hot encoding.

        Parameters:
            data (DataFrame): DataFrame containing the CSV data.

        Returns:
            DataFrame: DataFrame with categorical features encoded using one-hot encoding.
        """
        # Get list of object columns (categorical features)
        cat_columns = data.select_dtypes(include=['object', 'bool']).columns
        print(cat_columns)

        # Perform one-hot encoding
        data_encoded = pd.get_dummies(data, columns=cat_columns, dtype=int)

        logging.info("Data dimensions after encoding categorical features: %s", data_encoded.shape)
        return data_encoded

    def scale_data(self, data, method='minmax', applied=False):
        """
        Apply scaling to the encoded data.

        Parameters:
            data (DataFrame): DataFrame containing the encoded data.
            method (str): Scaling method to use: 'minmax' for Min-Max Scaling or 'standard' for Standard Scaling.
                Default is 'minmax'.
            applied (bool): Whether to apply scaling or not. If False, the function will return the original data without scaling.

        Returns:
            DataFrame: DataFrame with scaled features if applied is True, otherwise returns the original data.
        """
        if not applied:
            logging.warning("Scaling of features was skipped.")
            return data

        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        else:
            logging.warning("Invalid scaling method. Using Min-Max Scaling by default.")
            scaler = MinMaxScaler()

        scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        logging.info("Data dimensions after scaling: %s", scaled_data.shape)
        return scaled_data

    def split_data(self, data):
        """
        Split the data into training and testing sets.

        Parameters:
            data (DataFrame): DataFrame containing the CSV data.

        Returns:
            tuple: Tuple containing train and test DataFrames.
        """
        # Extract the latest month for testing
        latest_month = data['timeslot_datetime_from'].max().month
        latest_year = data['timeslot_datetime_from'].max().year
        test_data = data[(data['timeslot_datetime_from'].dt.month == latest_month) &
                         (data['timeslot_datetime_from'].dt.year == latest_year)]

        # Use the remaining data for training
        train_data = data[data['timeslot_datetime_from'] < test_data['timeslot_datetime_from'].min()]
        print("Dimension of training data:", train_data.shape)
        print("Dimension of testing data:", test_data.shape)

        return train_data, test_data

    def train_model(self, train_data):
        """
        Train the regression model.

        Parameters:
            train_data (DataFrame): DataFrame containing the training data.

        Returns:
            model: Trained regression model.
        """
        # Select features and target variable
        X = train_data.drop(columns=['share_15_54', 'timeslot_datetime_from'])
        y = train_data['share_15_54']

        # Choose a regression model
        # We could use also other models like Gradient Boosting Regressor, XGBoost regressor, etc.
        # model = RandomForestRegressor()
        model = GradientBoostingRegressor()
        # model = xgb.XGBClassifier()

        # Train the model
        model.fit(X, y)
        return model

    def evaluate_model(self, test_data, test_predictions):
        """
        Evaluate the performance of the regression model.

        Parameters:
            test_data (DataFrame): DataFrame containing the testing data.
            test_predictions (array-like): Predicted values for the testing data.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        # Append 'timeslot_datetime_from' column back to the test data
        test_data_with_predictions = test_data.copy()
        test_data_with_predictions['timeslot_datetime_from'] = test_data['timeslot_datetime_from']

        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(test_data['share_15_54'], test_predictions)

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(test_data['share_15_54'], test_predictions)

        # Calculate Root Mean Squared Error (RMSE)
        rmse = root_mean_squared_error(test_data['share_15_54'], test_predictions)

        # Calculate R-squared (R^2) score
        r2 = r2_score(test_data['share_15_54'], test_predictions)

        # Create a dictionary to store the evaluation metrics
        evaluation_metrics = {
            'Mean Squared Error (MSE)': mse,
            'Mean Absolute Error (MAE)': mae,
            'Root Mean Squared Error (RMSE)': rmse,
            'R-squared (R^2) Score': r2
        }

        return evaluation_metrics, test_data_with_predictions

def main(csv_file):
    # Set up logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Create an instance of the SharePredictionModel class with the CSV file path
    model = SharePredictionModel('data_1.csv')
    # Load CSV data and log the dimensions of the DataFrame
    data = model.load_data()
    # Process the loaded data
    processed_data = model.process_data(data, duplicate_rows=True)
    # Training the model with or without a specific features
    selected_features = model.select_features(data=processed_data, include_main_ident=False,
                                              include_share_15_54_3mo_mean=False)
    # Extract datetime features
    data_with_datetime_features = model.extract_datetime_features(data=selected_features, enable=True)
    # Encode categorical features
    data_encoded = model.encode_categorical_features(data=data_with_datetime_features)
    # Apply Scaling to the encoded data
    scaled_data_minmax = model.scale_data(data_encoded, method='minmax', applied=False)
    # Split the data into training and testing sets
    train_data, test_data = model.split_data(data=data_encoded)
    # Train the regression model
    regression_model = model.train_model(train_data=train_data)
    # Make predictions on the test data
    test_predictions = regression_model.predict(test_data.drop(columns=['share_15_54', 'timeslot_datetime_from']))
    # Evaluate model performance
    evaluation_metrics, test_data_with_predictions = model.evaluate_model(test_data=test_data,
                                                                          test_predictions=test_predictions)
    # Print evaluation results
    print("Evaluation Metrics:")
    for metric, value in evaluation_metrics.items():
        print(f"{metric}: {value}")
    # Create a DataFrame to compare predicted and real values
    # comparison_df = pd.DataFrame({
    #     'timeslot_datetime_from': test_data_with_predictions['timeslot_datetime_from'],  # Timestamp column
    #     'real_share_15_54': test_data_with_predictions['share_15_54'],  # Real values
    #     'predicted_share_15_54': test_predictions  # Predicted values
    # })

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Train a share prediction model using the provided CSV file.")
    parser.add_argument("csv_file", type=str, help="Path to the CSV file containing the data.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Run the main function with the provided CSV file path
    main(args.csv_file)