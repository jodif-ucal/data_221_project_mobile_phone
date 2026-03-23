import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#Importing the data from the csv file
mobile_prices_df = pd.read_csv("Data/mobile_phone_dataset.csv")

#Splitting the dataset between all the features and the target labels
feature_matrix = mobile_prices_df.drop(columns="price_range")
labels = mobile_prices_df["price_range"]

#Splitting the data into a training and testing set
features_train, features_test, labels_train, labels_test = train_test_split(
    feature_matrix, labels, test_size=0.25, random_state=42
)


#Standardising the matrix
#Looping through each column
for column in feature_matrix:

    #Checking what the maximum value of a column is --- if it's greater than 1 (non-binary feature) then
    # standardise the column
    if feature_matrix[column].max() > 1:

        #Getting the mean of the column
        mu = feature_matrix[column].mean()

        #Getting the standard deviation of the column
        sigma = feature_matrix[column].std()

        #Standardising the column using ((column - mean) / standard deviation)
        feature_matrix[column] = (feature_matrix[column] - mu) / sigma

