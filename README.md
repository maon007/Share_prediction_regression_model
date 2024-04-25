# Share Prediction Model

This project is split into 3 parts and aims to:
1. analyze a dataset containing information about television channels, timeslots, and movie IDs. The objective of this part is to run a general analysis and draw conclusions about the channels based on the provided data.
2. building a predictive model to estimate the audience share for a specific demographic ("share 15 54") without using the three-month rolling mean of the audience share. The focus is on utilizing features and the timestamp to make accurate predictions.
3. leverage the mean of the audience share ("share 15 54 3mo mean") to enhance predictions for the audience share. This task allows the creation of new features and the use of black-box models that can be explained to some extent. The choice of model should consider factors such as cost, interpretability, and performance.

## Overview

The project consists of the following main components:

1. **General Data Exploration:**
    - Displaying statistics about features (data types, dimensions, etc.)
    - Checking the number of NULL values
    - Checking the number of empty columns
    - Checking duplicated rows
    - Calculation of correlation matrix

2. **Features Exploration and Analysis:**
    - Displaying viewership trends
    - Visualizing viewer share distribution
    - Analysis of genres
    - Analysis of movie origins
    
3. **Data Loading and Preprocessing:**
   - Loads data from a CSV file.
   - Handles missing values and duplicates.
   - Selects and encodes features as necessary.

2. **Feature Engineering:**
   - Extracts datetime features from the `timeslot_datetime_from` column:
     - Day of the week
     - Month of the year
     - Hour of the day
     - Season
    - Scaling the data

3. **Model Training and Evaluation:**
   - Trains and evaluates the ML model.
   - Uses metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R^2) score for evaluation.

4. **Dockerization (Optional):**
   - Includes Dockerfiles to containerize the environment and facilitate deployment (optional).

## Attached files
- **data_exploration_and_analysis.py**: This script runs the general exploration and analyzes channels more in detail (visualization, etc.)
- **share_prediction.py**: This script prepares the data, trains, and evaluates the ML model
- **tv_nova_exploration.ipynb**: This Notebook works as a preparation for the analysis of the Python file
- **tv_nova_model.ipynb**: This Notebook works as a preparation for the prediction Python file
- **requirements_analysis.txt**: This file contains all the Python libraries used for data_exploration_and_analysis.py 
- **requirements_model.txt**: This file contains all the Python libraries used for share_prediction.py
- **Dockerfile_analysis**: This file contains instructions for building your Docker image of the analysis part.
- **Dockerfile_prediction**: This file contains instructions for building your Docker image of the prediction part.


## Usage

### Direct way

1. Clone the repository:
   ```bash
   git clone https://github.com/maon007/TV_nova_share_prediction_model.git
   ```
2. Install dependencies:
    - for the analysis part:
   ```bash
    pip install -r requirements_analysis.txt
   ```
   - for the model prediction part:
   ```bash
    pip install -r requirements_model.txt
   ```

#### Running the Analysis and Model
To analyze the data, train the model, and make predictions, follow these steps:
- Ensure you have a CSV file containing the data (e.g., data_1.csv).
- Run the main Python script with the path to the CSV file as an argument (e.g.):
    - **for the analysis:**
   ```bash
    python data_exploration_and_analysis.py "./data_1.csv"
   ```
- The scripts will load the data, run the general exploration and analysis of the data, analyze channels more in detail, and return information from the analysis

    - **for the prediction:**
   ```bash
    python share_prediction.py "./data_1.csv"
   ```
- The scripts will load the data, preprocess it, train the model, make predictions, evaluate the model's performance, and return essential information.

### Using Dockerfiles
You can also use a Dockerfile to build the image for the scripts. Open a terminal and navigate to the directory containing Dockerfiles and Python files. Build the image using the following command:
- **for the analysis:**
```bash
docker build -t my_docker_image1 -f Dockerfile_analysis .
```
You can replace "my_docker_image1" with the desired name for your Docker image.

Run the Docker container: Once the image is built, you can run it as a container:
```bash
docker run my_docker_image1
```
- **for the prediction:**
```bash
docker build -t my_docker_image2 -f Dockerfile_prediction .
```
You can replace "my_docker_image2" with the desired name for your Docker image.

Run the Docker container: Once the image is built, you can run it as a container:
```bash
docker run my_docker_image2
```

**NOTE:** Both Docker Images were tested and run successfully.


