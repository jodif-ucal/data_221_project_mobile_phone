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


