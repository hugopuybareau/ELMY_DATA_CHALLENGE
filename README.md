# ELMY Data Challenge

This project is my submission for the ELMY Data Challenge, aimed at predicting the price difference between the SPOT and Intraday electricity markets using various Machine Learning and modeling techniques.

# Project Objectives
The primary goal was to predict the spot_id_delta variable, which represents the difference between the prices of the two markets. To achieve this, I employed several techniques for preprocessing, modeling, and evaluation, addressing challenges such as handling outliers, missing data, and temporal fluctuations in the dataset.

# Main Steps

## 1. Data Preprocessing

Data Cleaning: Managed missing values (NaN) in key columns such as solar_power_forecasts_std, predicted_spot_price, and load_forecast. Techniques such as mean imputation, forward fill (using the previous value), and model-based imputations were applied.
Outlier Handling: Identified outliers using the Z-score method and managed them appropriately to avoid biases in predictions.
Temporal Feature Engineering: Extracted new features from the DELIVERY_START column, such as hour of the day, day of the week, month, and whether it's a weekend or a peak hour, to capture temporal patterns in the data.

## 2. Model Selection and Training
   
Modeling: I explored several machine learning algorithms, including RandomForestRegressor and XGBoost, to build robust predictive models.
Hyperparameter Tuning: Utilized GridSearchCV and RandomizedSearchCV to optimize the models' hyperparameters and improve their performance.
Neural Networks: Implemented a simple feedforward neural network for regression tasks using TensorFlow and Keras, though performance evaluation showed better results with ensemble models.

## 3. Evaluation Metrics
   
Weighted Accuracy: Given the challenge's focus on predicting the direction of the price change, I used Weighted Accuracy as a key evaluation metric. This metric emphasizes the correct prediction of the sign of spot_id_delta, with higher weights on larger differences.
Mean Squared Error (MSE): To assess the overall regression accuracy, I used MSE, aiming for values as close to 0 as possible.

## 4. Handling Differences Between Training and Submission Data
   
Correlation Analysis: I compared the correlations between the training and submission datasets to identify significant differences and adapted feature transformations accordingly.
Normalization: Features were normalized using a StandardScaler, fitted on the training data, to improve model performance on both the training and submission datasets.

## 5. Final Submission
   
Prediction Pipeline: The final pipeline used an optimized XGBoost model to predict spot_id_delta for the submission dataset, after handling missing values and normalizing features.
CSV Export: The predicted values were saved in a .csv file, ready for submission to the challenge.
