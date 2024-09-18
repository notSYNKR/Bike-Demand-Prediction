# Bike-Demand-Prediction

## Synopsis

The goal of this project is to create a machine-learning model capable of predicting the number of bike rentals based on various factors, including the date, weather conditions, and time of day. The project uses two datasets: one containing daily summaries of bike rentals (the "day" dataset) and another with detailed hourly data (the "hour" dataset).

### Dataset Overview
Day Dataset: This dataset provides a high-level, daily aggregate of bike rentals, summarizing the total rentals per day and relevant variables such as weather, season, and holiday information.
Hour Dataset: In contrast, the hour dataset breaks down bike rentals by the hour, giving a more granular view of how factors like temperature and weather affect demand throughout the day

## Folder structure

```
├───artifacts
├───logs
├───notebook
│   └───data
├───src
│   ├───components
│   ├───pipeline
├───static
├───templates
```
- `notebook/`: This directory contains Jupyter notebooks used for tasks like data analysis, preprocessing, and developing machine learning models.

- `notebook/data/`: Stores the dataset files used in the project.

- `src/`: Houses the core project code, including scripts for data preprocessing and model training.

- `src/pipeline/`: Contains code for the various machine learning pipelines that have been implemented.

- `artifacts/`: A folder for storing trained models, evaluation metrics, and prediction outputs.

- `static/ and templates/`: These folders contain the frontend components required for deploying the project via a Flask web application.

## The "day" dataset consists of the following columns:


```
instant: A unique identifier assigned to each record.
dteday: The date associated with the record.
season: The season during which the data was recorded (1 = Spring, 2 = Summer, 3 = Fall, 4 = Winter).
yr: The year the data was collected (0 = 2011, 1 = 2012).
mnth: The month of the year (1 = January to 12 = December).
holiday: A binary indicator specifying whether the day is a holiday (1 = Yes, 0 = No).
weekday: The day of the week, where 0 represents Sunday and 6 represents Saturday.
workingday: A binary indicator signifying if the day is a working day (1 = Yes, 0 = No).
weathersit: The weather condition at the time of recording (1 = Clear, 2 = Mist/Cloudy, 3 = Light Rain/Snow, 4 = Heavy Rain/Snow).
temp: Normalized temperature measured in Celsius.
atemp: Normalized "feels like" temperature in Celsius.
hum: Normalized humidity level.
windspeed: Normalized wind speed.
casual: The number of bike rentals by casual, non-registered users.
registered: The number of bike rentals by registered users.
cnt: The total number of bike rentals, combining both casual and registered users.
```


## Hour Dataset

The "hour" dataset contains the following columns:

    instant: A unique identifier for each record.
    dteday: The date on which the data was recorded.
    season: The season of the year (1 = Spring, 2 = Summer, 3 = Fall, 4 = Winter).
    yr: The year of the data (0 = 2011, 1 = 2012).
    mnth: The month of the year, ranging from 1 (January) to 12 (December).
    hr: The hour of the day (from 0 to 23).
    holiday: A binary indicator showing whether the day is a holiday (1 = Yes, 0 = No).
    weekday: The day of the week, where 0 = Sunday and 6 = Saturday.
    workingday: A binary indicator showing if the day is a working day (1 = Yes, 0 = No).
    weathersit: The weather condition (1 = Clear, 2 = Mist/Cloudy, 3 = Light Rain/Snow, 4 = Heavy Rain/Snow).
    temp: The normalized temperature in Celsius.
    atemp: The normalized "feels like" temperature in Celsius.
    hum: The normalized humidity level.
    windspeed: The normalized wind speed.
    casual: The number of bike rentals by non-registered users.
    registered: The number of bike rentals by registered users.
    cnt: The total number of bike rentals, including both casual and registered users.

    
To run the project locally, please ensure you have the following dependencies installed:

- Python 3.7 or higher
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook 
- ipykernel


Once you have the dependencies, follow these steps to set up the project:

1. Clone the repository: `git clone https://github.com/notSYNKR/Bike-Demand-Prediction.git`
3. Navigate to the project directory: `cd Bike-Demand-Prediction`
4. Create a virtual environment (optional): `conda create -p venv python==3.8`
5. Activate the virtual environment (optional): `activate venv/`
6. Install the required packages: `pip install -r requirements.txt`

## Results

The bike demand predictions are assessed using several performance metrics, including mean absolute error (MAE), root mean squared error (RMSE), and the R-squared score. These metrics provide valuable insights into the model's accuracy and effectiveness in predicting bike rentals:

    MAE: Represents the average absolute difference between predicted and actual values, indicating overall prediction accuracy.
    RMSE: Measures the square root of the average squared differences between predicted and actual values, emphasizing larger errors.
    R-squared: Indicates the proportion of variance in the actual data that is explained by the model, with values closer to 1 showing a better fit.


## Model Building and Selection
To forecast the number of bike rentals, various machine-learning models were implemented and evaluated for their performance. Each model was assessed using R-squared metrics to determine how well they fit the data. The models used include:

    Linear Regression: A simple yet powerful algorithm that models the relationship between the input features and the bike rental count by fitting a straight line through the data points.
    R-squared: 38.45
    
    Random Forest: An ensemble learning method that builds multiple decision trees during training and outputs the average prediction, reducing overfitting and improving accuracy.
    R-squared: 93.69
    
    Extra Trees Regressor: Similar to Random Forest, but it selects cut points at random, leading to a more diversified set of trees and better generalization.
    R-squared: 94.31
    
    LightGBM: A gradient boosting framework that is highly efficient and fast, designed for large datasets with high-dimensional features, often yielding excellent predictive performance.
    R-squared: 94.38
    
    XGBoost: Another gradient boosting method that focuses on optimizing speed and performance, often delivering top results in many machine learning tasks.
    R-squared: 94.35


After training and evaluating these models, XGBoost was chosen as the final model due to its superior performance in terms of accuracy and predictive power.


To use the deployed model with Flask, follow these steps:

- `Install the necessary dependencies by running pip install -r requirements.txt (ensure any commented lines are removed).`
- `Train the model by executing python src/pipeline/training_pipeline.py, which generates the trained model.`
- `Launch the Flask application by running python app.py.`
- `Open the application in your web browser using the provided URL.`
- `Input the required features, such as date, weather conditions, and time.`
- `Click the "Predict" button to generate the predicted bike rental count.`


Please note that the accuracy of the predictions may vary based on the input data and model performance.


## Conclusion

This project showcases the effective use of machine learning techniques to predict bike rental volumes. By employing the LightGBM model and integrating diverse input features, the predictions are both precise and reliable. This enables better decision-making for managing bike rental operations.

For additional information, including code implementation and comprehensive analysis, please visit the code repository.
