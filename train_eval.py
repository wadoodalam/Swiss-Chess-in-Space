'''
Author: Wadood Alam
Date: 25th January 2024
Class: AI 539
Assignment: Homework1 - Swiss Cheese in Space

'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer, KNNImputer


def LoadData():
    # missing values are represented by 99/-99 in the dataset
    data = pd.read_csv('cfhtlens.csv', na_values=[99,-99], usecols=['CLASS_STAR','PSF_e1','PSF_e2','scalelength','model_flux','MAG_u','MAG_g','MAG_r','MAG_i','MAG_z'])
    # Assign the relevant columns/data to X  
    X = data[['PSF_e1','PSF_e2','scalelength','model_flux','MAG_u','MAG_g','MAG_r','MAG_i','MAG_z']]
    # Assign class_star 0 or 1 to Y depending on the condition
    Y = (data['CLASS_STAR'] >= 0.5).astype(int)
    return X,Y

def TrainTestSplit(X,Y, omitFeatures=False):
    if omitFeatures:
        # drop col with missing values
        X = X.dropna(axis=1,how='any')
        
    # Identify rows with atleast 1 missing value
    missing = np.isnan(X).any(axis=1)
    # The test data does not include rows with missing values
    X_train, X_test, Y_train, Y_test = train_test_split(X[~missing], Y[~missing], train_size=3000, random_state=0)
    

    # Concatenate the missing value rows in the test set
    X_test = pd.concat([X_test, X[missing]])
    Y_test = pd.concat([Y_test, Y[missing]])

    return X_train, X_test, Y_train, Y_test


def TrainRandomForest(X_train, Y_train):
    # call the random forest classifier with parameters
    classifier = RandomForestClassifier(random_state=0,min_samples_split=10,n_estimators=100)
    # train the model
    classifier.fit(X_train, Y_train)
    #print('Parameters of RF:', classifier.get_params())
    
    return classifier


def Predict(strategy):
    # Load the data using this function
    X,Y = LoadData()
    # Get the test and train set
    X_train, X_test, Y_train, Y_test = TrainTestSplit(X,Y)
    # Train the model using this function
    classifier = TrainRandomForest(X_train,Y_train)
     # Identify rows with atleast 1 missing value
    missing = np.isnan(X_test).any(axis=1)

    if strategy == 'abstain':
        # Predict using X_test dataset without the missing values
        Y_predict = classifier.predict(X_test[~missing])

        # Create a new np array, add the predicted values and 99(none) in correspondence to missing items
        Y_predict_filled = np.full(Y_test.shape, np.nan)
        Y_predict_filled[~missing] = Y_predict
        Y_predict_filled[missing] = 99

        # Calculate both accuracies
        accuracy = accuracy_score(Y_predict_filled,Y_test)
        # abstaining counts as error hence 0 accuracy for missing data points
        missing_accuracy = 0
        print('Number of items missing are =', np.sum(missing))
    



    if strategy == 'majorityClass':
        # Get the mode from train set, use the first value in the array
        mode_class = Y_train.mode().iloc[0]
        
        print("Num of 0's in the entire test set =",np.sum(Y_test == 0)) 
        # Create and fill in the predict np array using that mode
        Y_predict = np.full(Y_test.shape, mode_class)
        c = 0
        for pred,actual in zip(Y_predict,Y_test[missing]):
            if pred == actual:
                c+=1
        print("number of 0's for the missing points in dataset(824) =",c)
        '''
        There are a total of 1202 0's in class_star test set and 824 with missing points. Since the majority class from train set predicts 0,
        the accuracy for the entire test set will be  1202/2000 = 0.601
        Out of the 824 missing points in the test set, the class_star has value of 0 336 times.
        the accuracy for missing points in the test set will be 336/824 = 0.4077
        '''
        # Calculate both accuracies
        accuracy = accuracy_score(Y_test, Y_predict)
        missing_accuracy = accuracy_score(Y_test[missing], Y_predict[missing])
    
    
    
    
    
    
    if strategy == 'omitFeatures':
        # with omitFeatures set to true, getting the split accordingly
        X_train_omit, X_test_omit, Y_train_omit, Y_test_omit = TrainTestSplit(X,Y,omitFeatures=True)

        #Re-train the classifier
        classifier = TrainRandomForest(X_train_omit,Y_train_omit)
        Y_predict = classifier.predict(X_test_omit)

        # need to convert to numpy, and treat as type int due to misalignment issues due to dropping features
        Y_test_omit = Y_test_omit.to_numpy()
        Y_predict = Y_predict.astype(int)

        # Calculate both accuracies
        missing_accuracy = accuracy_score(Y_test_omit[missing], Y_predict[missing])
        accuracy = accuracy_score(Y_test_omit, Y_predict)
    
    if strategy == 'imputeAvg':
        # Impute with mean strategy 
        imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
        imputer.fit(X_train)
        imputed_data = imputer.transform(X_test)
        
        # Predict after imputation
        Y_predict = classifier.predict(imputed_data)

        # Calculate both accuracies
        missing_accuracy = accuracy_score(Y_test[missing], Y_predict[missing])
        accuracy = accuracy_score(Y_test,Y_predict)

    if strategy == 'imputeNeighbor':
         # Impute with k-nearest neighbor strategy 
        imputer = KNNImputer(n_neighbors=5)   
        imputer.fit(X_train)
        imputed_data = imputer.transform(X_test)

        # Predict after imputation
        Y_predict = classifier.predict(imputed_data)
        
        # Calculate both accuracies
        missing_accuracy = accuracy_score(Y_test[missing], Y_predict[missing])
        accuracy = accuracy_score(Y_test,Y_predict)

    # 4 d.p format 
    formatted_accuracy = "{:.4f}".format(accuracy)
    formatted_missing_accuracy = "{:.4f}".format(missing_accuracy)

    return formatted_accuracy, formatted_missing_accuracy



if __name__ == "__main__":
    
    # Types of strategies
    strategies = ['abstain','majorityClass','imputeAvg','imputeNeighbor', 'omitFeatures']
    # List to stores accuracies and their corresponding accuracies
    strategyAccuracy = []
    for strategy in strategies: 
        accuracy,missing_accuracy = Predict(strategy)
        # Append accuracies and strategies to the list
        strategyAccuracy.append((strategy,accuracy,missing_accuracy))
    print(strategyAccuracy)

    # Load data, split in train-test, train classifier
    X,Y = LoadData()
    X_train, X_test, Y_train, Y_test = TrainTestSplit(X,Y)
    classifier = TrainRandomForest(X_train,Y_train)
    

    # No missing data for my query result
    feature_vector = np.array([ 0.0096, -0.0036, 0, 30.9, 26.2125, 23.7822, 22.6081, 21.9143, 21.6]).reshape(1, -1)
    # Predict using feature vector
    prediction = classifier.predict(feature_vector)
    # Print the prediction
    if prediction == 1:
        print("It is a Galaxy")
    elif prediction == 0:
         print("It is a Star")

