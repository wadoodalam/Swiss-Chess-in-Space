import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def Graph(data):
    data.hist()
    plt.savefig('histogram.png')

def Describe(data):
    description = data.describe()
    description = description.loc[['min', 'max', 'mean', '50%','count']]
    description.to_csv('profile.csv')



if __name__ == "__main__":

    # missing values are represented by 99/-99 in the dataset
    data = pd.read_csv('cfhtlens.csv', na_values=[99,-99])

    Describe(data) # mean, median, max, min, num_of_missing values
    #Graph(data) # Create Histogram
    