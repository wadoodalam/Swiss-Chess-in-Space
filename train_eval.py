import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def LoadData():
    # missing values are represented by 99/-99 in the dataset
    data = pd.read_csv('cfhtlens.csv', na_values=[99,-99], usecols=['CLASS_STAR','PSF_e1','PSF_e2','scalelength','model_flux','MAG_u','MAG_g','MAG_r','MAG_i','MAG_z'])
   
    X = data[['PSF_e1','PSF_e2','scalelength','model_flux','MAG_u','MAG_g','MAG_r','MAG_i','MAG_z']]
    Y = (data['CLASS_STAR'] >= 0.5).astype(int)
    return X,Y

def TrainTestSplit(X,Y):
    missing = np.isnan(X).any(axis=1)
    # The test data does not include rows with missing values
    X_train, X_test, Y_train, Y_test = train_test_split(X[~missing], Y[~missing], train_size=3000, random_state=0)
    #print("Training set count:", len(Y_train))

    # Concatenate the missing value rows in the test set
    X_test_missing_values, Y_test_missing_values = X[missing], Y[missing]
    X_test = pd.concat([X_test, X_test_missing_values])
    Y_test = pd.concat([Y_test, Y_test_missing_values])

    return X_train, X_test, Y_train, Y_test

def TrainRandomForest(X_train, Y_train):
    classifier = RandomForestClassifier(random_state=0)
    classifier.fit(X_train, Y_train)
    return classifier




if __name__ == "__main__":

    X,Y = LoadData()
    X_train, X_test, Y_train, Y_test = TrainTestSplit(X,Y)
    classifier = TrainRandomForest(X_train,Y_train)
    
    