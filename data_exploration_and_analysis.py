# libraries
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

TASK_1_CSV_1 = "./data_1.csv"
TASK_1_CSV_2 = "./data_2.csv"


def load_csv_to_dataframe(file_path):
    """
    Load data from a CSV file into a pandas DataFrame.

    Parameters:
    - file_path (str): The path to the CSV file.

    Returns:
    - DataFrame: A pandas DataFrame containing the data from the CSV file.
    """
    try:
        # Load the CSV file into a DataFrame
        # low_memory - handling mixed data type columns (non-numeric stored as an object)
        df = pd.read_csv(file_path,
                         low_memory=False)
        print("Data dimensions after loading the csv file", df.shape)
        return df
    except FileNotFoundError:
        print("File not found. Please provide a valid file path.")
        return None


def sort_dataframe(df):
    """
    Sort a pandas DataFrame by channel_id, main_indent, and timeslot_datetime_from in ascending order.
    """
    sorted_df = df.sort_values(by=['channel_id', 'main_ident', 'timeslot_datetime_from'])
    return sorted_df


### GENERAL EXPLORATION PART ###
def statistics(df, describe=True, unique_count=False, missing_count=False, data_types=True):
    """
    Display statistics for a pandas DataFrame including count of unique values, count of missing values,
    and data types of columns.

    Parameters:
    - df (DataFrame): The pandas DataFrame for which extended statistics are to be calculated.
    """
    try:
        if describe:
            # Basic statistics for numerical columns
            print("Basic statistics for numerical columns:")
            print(df.describe())

        if unique_count:
            # Count of unique values for each column
            print("\nCount of unique values for each column:")
            print(df.nunique())

        if missing_count:
            # Count of missing values for each column
            print("\nCount of missing values for each column:")
            print(df.isnull().sum())

        if data_types:
            # Data types of columns
            print("\nData types of columns:")
            print(df.dtypes)
    except Exception as e:
        print("An error occurred:", e)


def assign_data_types(df):
    """
    Convert timeslot_datetime_from to timestamp format
    """
    df['timeslot_datetime_from'] = pd.to_datetime(df['timeslot_datetime_from'])
    return df


def find_duplicate_rows(df):
    """
    Find duplicate rows in a pandas DataFrame.

    Parameters:
    - df (DataFrame): The pandas DataFrame to be checked for duplicate rows.

    Returns:
    - DataFrame: The DataFrame with duplicate rows removed.
    """
    try:
        # Check for duplicate rows
        duplicate_rows = df[df.duplicated()]

        # If duplicate rows are found
        if not duplicate_rows.empty:
            num_duplicates = len(duplicate_rows)
            total_rows = len(df)
            duplicate_percentage = (num_duplicates / total_rows) * 100

            print("Number of duplicate rows found:", num_duplicates)
            print("Percentage of duplicate rows from the total:", duplicate_percentage, "%")

            return df
        else:
            print("No duplicate rows found.")
            return df
    except Exception as e:
        print("An error occurred:", e)
        return None


def null_percentage(df, just_nulls=False):
    """
    Calculate and display the percentage of null values for each column in a pandas DataFrame.

    Parameters:
    - df (DataFrame): The pandas DataFrame for which null percentages are to be calculated.
    - just_nulls (bool): If True, show only columns with all null values.
    """
    try:
        # Calculate the percentage of null values for each column
        null_percentages = (df.isnull().sum() / len(df)) * 100

        if just_nulls:
            # Filter columns with all null values
            null_columns = null_percentages[null_percentages == 100]
            if not null_columns.empty:
                print("Columns with all null values:")
                print(null_columns)
            else:
                print("No columns with all null values found.")
        else:
            # Display the percentage of null values for each column
            print("Percentage of null values for each column:")
            print(null_percentages)
    except Exception as e:
        print("An error occurred:", type(e).__name__, "-", e)


def count_rows_with_null(df):
    """
    Calculate the number of rows containing null values in a pandas DataFrame.

    Parameters:
    - df (DataFrame): The pandas DataFrame to be checked for null values.

    Returns:
    - int: The number of rows containing null values.
    """
    print("Total number of rows:", df.shape[0])
    # Drop columns containing just NULL values
    df = df.dropna(axis=1, how='all')
    # Calculate the number of rows containing null values
    rows_with_null = df.isnull().any(axis=1).sum()
    print("Number of rows containing null values after dropping empty columns:", rows_with_null)
    return rows_with_null


def visualize_correlation(df):
    """
    This function calculates the correlation matrix for numerical columns in the DataFrame
    and visualizes it using a heatmap.

    Args:
        df: A pandas DataFrame containing numerical columns.

    Returns:
        The correlation matrix heatmap is displayed directly.
    """
    # Filter out only numerical columns
    df = df.dropna(axis=1, how='all')
    numerical_df = df.select_dtypes(include=['number'])

    # Calculate correlation matrix
    correlation_matrix = numerical_df.corr()

    # Visualize correlations using heatmap
    plt.figure(figsize=(17, 17))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()


### EXPLORATION OF FEATURES AND CONCLUSIONS ###

# Viewership Trends
def calculate_viewer_share_stats(df):
    """
    Calculate descriptive statistics of viewer share for each channel.
    You can use these statistics to gain insights into the distribution of viewer share across different channels.
    For example, you can identify:
        - channels with the highest average viewer share,
        - channels with the most variability in viewer share, or
        - channels with the highest and lowest viewer share ranges.
        - prumerna sledovat pro kanal (kde nejvetsi, kde nejmensi)

    Parameters:
    - df (DataFrame): The pandas DataFrame containing the data.

    Returns:
    - DataFrame: Descriptive statistics of viewer share for each channel.
    """
    try:
        viewer_share_stats = df.groupby('channel_id')['share_15_54'].describe()
        print('viewer_share_stats:', viewer_share_stats)
        return viewer_share_stats
    except Exception as e:
        print("An error occurred:", type(e).__name__, "-", e)


def plot_viewer_share_distribution(data):
    """
    This function visualizes the distribution of viewer shares across timeslots
    for all four channels using Matplotlib and Seaborn with different colors for each channel.

    Args:
        data: A pandas DataFrame containing the channel data.

    Returns:
        None. The plot is displayed directly.
    """
    # Define a custom color palette with different colors for each channel
    channel_colors = sns.color_palette("husl", n_colors=len(data['channel_id'].unique()))

    # Initialize the matplotlib figure and axis
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    # Create scatter plots for each channel using Seaborn and custom colors
    sns.scatterplot(data=data, x='timeslot_datetime_from', y='share_15_54', hue='channel_id', palette=channel_colors,
                    ax=ax)

    # Set plot title and axis labels
    plt.title('Distribution of Viewer Shares (15-54) Across Timeslots')
    plt.xlabel('Timeslot Datetime From')
    plt.ylabel('Viewer Share (15-54)')

    # Show legend
    plt.legend(title='Channel')

    # Show plot
    plt.show()


def plot_genre_share(data):
    """
    This function visualizes the genres with the highest average share_15_54 (podil sledovanosti) and the number of movies per genre
    for channel_id = 3.

    Args:
        data: A pandas DataFrame containing the channel data.

    Returns:
        None. The plot is displayed directly.
    """
    # Filter data for channel_id = 3
    channel_3_data = data[data['channel_id'] == 3]

    # Group data by genre and calculate the average share_15_54 for each genre
    genre_stats = channel_3_data.groupby('ch3__f_10').agg({'share_15_54': 'mean', 'main_ident': 'count'}).reset_index()

    # Sort data by average share_15_54 in descending order
    sorted_data = genre_stats.sort_values(by='share_15_54', ascending=False)

    # Initialize the matplotlib figure and axis
    plt.figure(figsize=(12, 6))

    # Create bar plot for average share_15_54
    plt.subplot(1, 2, 1)
    sns.barplot(data=sorted_data, x='ch3__f_10', y='share_15_54', hue='ch3__f_10', palette='viridis', legend=False)
    plt.title('Top Genres by Average Viewer Share (15-54) for ch3')
    plt.xlabel('Genres')
    plt.ylabel('Average Viewer Share (15-54)')
    plt.xticks(rotation=45, ha='right')

    # Create bar plot for number of movies
    plt.subplot(1, 2, 2)
    sns.barplot(data=sorted_data, x='ch3__f_10', y='main_ident', hue='ch3__f_10', palette='viridis', legend=False)
    plt.title('Number of Movies per Genre for ch3')
    plt.xlabel('Genres')
    plt.ylabel('Number of Movies')
    plt.xticks(rotation=45, ha='right')

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()


def plot_country_share(data):
    """
    This function visualizes the countries with the highest average share_15_54 (podil sledovanosti) and the number of movies per country
    for channel_id = 3.

    Args:
        data: A pandas DataFrame containing the channel data.

    Returns:
        None. The plot is displayed directly.
    """
    # Filter data for channel_id = 3
    channel_3_data = data[data['channel_id'] == 3]

    # Group data by country and calculate the average share_15_54 and count of movies for each country
    country_stats = channel_3_data.groupby('ch3__f_11').agg(
        {'share_15_54': 'mean', 'main_ident': 'count'}).reset_index()

    # Sort data by average share_15_54 in descending order
    sorted_data = country_stats.sort_values(by='share_15_54', ascending=False)

    # Initialize the matplotlib figure and axis
    plt.figure(figsize=(12, 6))

    # Create bar plot for average share_15_54
    plt.subplot(1, 2, 1)
    sns.barplot(data=sorted_data, x='ch3__f_11', y='share_15_54', hue='ch3__f_11', palette='viridis', legend=False)
    plt.title('Top Countries by Average Viewer Share (15-54) for ch3')
    plt.xlabel('Countries')
    plt.ylabel('Average Viewer Share (15-54)')
    plt.xticks(rotation=45, ha='right')

    # Create bar plot for number of movies
    plt.subplot(1, 2, 2)
    sns.barplot(data=sorted_data, x='ch3__f_11', y='main_ident', hue='ch3__f_11', palette='viridis', legend=False)
    plt.title('Number of Movies per Country for ch3')
    plt.xlabel('Countries')
    plt.ylabel('Number of Movies')
    plt.xticks(rotation=45, ha='right')

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <path_to_csv_file>")
        sys.exit(1)

    csv_file_path = sys.argv[1]

    # Load CSV file to DataFrame
    df = load_csv_to_dataframe(csv_file_path)

    if df is not None:
        # Basic analysis of the data
        df = sort_dataframe(df=df)
        statistics(df=df, describe=False, unique_count=True, missing_count=True, data_types=True)
        df = assign_data_types(df=df)
        df = find_duplicate_rows(df=df)
        null_percentage(df=df, just_nulls=True)
        count_rows_with_null(df=df)
        visualize_correlation(df)
        # Exploration of channels and other features
        calculate_viewer_share_stats(df=df)
        plot_viewer_share_distribution(data=df)
        plot_genre_share(data=df)
        plot_country_share(data=df)

        print(df.head(5))