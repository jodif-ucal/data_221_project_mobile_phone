import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#Importing the data from the csv file
mobile_prices_df = pd.read_csv("Data/mobile_phone_dataset.csv")

#Splitting the dataset between all the features and the target labels
feature_matrix = mobile_prices_df.drop(columns="price_range")
labels = mobile_prices_df["price_range"]

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

#Splitting the data into a training and testing set
features_train, features_test, labels_train, labels_test = train_test_split(
    feature_matrix, labels, test_size=0.25, random_state=42
)

#If this script is ran directly, the code in main will run
#This is to prevent the print statements from running whenever the other notebooks take the training
# and testing data from this script
def main():
    #Print the number of rows and columns from the training and testing features
    print("Shape of training set:", features_train.shape)
    print("Shape of testing set:", features_test.shape)

    #value_counts() will show the counts of each label
    #They are all perfectly each 500 across the dataset
    print("Occurrence of each label in the dataset", labels.value_counts())

    #The following commands will do the same thing for the training and testing set
    print("Occurrence of each label in the training set", np.unique(labels_train, return_counts=True))
    print("Occurrence of each label in the testing set", np.unique(labels_test, return_counts=True))

#This block of code ensures that main() runs is this script is directly ran, and not when imported
# as a module
if __name__ == "__main__":
    main()