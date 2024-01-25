'''
Author: Wadood Alam
Date: 25th January 2024
Class: AI 539
Assignment: Homework1 - Swiss Cheese in Space

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Graph(data):
    # Create histogram
    data.hist()
    # Save the histogram to this file name
    plt.savefig('histogram.png')

def Describe(data):
    description = data.describe()
    # Separate the desired columns
    description = description.loc[['min', 'max', 'mean', '50%','count']]
    # Save the profile to this file name
    description.to_csv('profile.csv')



if __name__ == "__main__":

    # missing values are represented by 99/-99 in the dataset
    data = pd.read_csv('cfhtlens.csv', na_values=[99,-99])

    Describe(data) # mean, median, max, min, num_of_missing values
    #Graph(data) # Create Histogram
    