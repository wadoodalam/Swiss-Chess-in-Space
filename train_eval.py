import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer, KNNImputer


def LoadData():
    # missing values are represented by 99/-99 in the dataset
    data = pd.read_csv('cfhtlens.csv', na_values=[99,-99], usecols=['CLASS_STAR','PSF_e1','PSF_e2','scalelength','model_flux','MAG_u','MAG_g','MAG_r','MAG_i','MAG_z'])
   
    X = data[['PSF_e1','PSF_e2','scalelength','model_flux','MAG_u','MAG_g','MAG_r','MAG_i','MAG_z']]
    Y = (data['CLASS_STAR'] >= 0.5).astype(int)
    return X,Y

def TrainTestSplit(X,Y, omitFeatures=False):
    if omitFeatures:
        # drop col with missing values
        X = X.dropna(axis=1,how='any')
        
      
    missing = np.isnan(X).any(axis=1)
    # The test data does not include rows with missing values
    X_train, X_test, Y_train, Y_test = train_test_split(X[~missing], Y[~missing], train_size=3000, random_state=0)
    

    # Concatenate the missing value rows in the test set
    X_test = pd.concat([X_test, X[missing]])
    Y_test = pd.concat([Y_test, Y[missing]])

    return X_train, X_test, Y_train, Y_test


def TrainRandomForest(X_train, Y_train):
    # random_state=0,min_samples_split=10,n_estimators=100
    classifier = RandomForestClassifier(random_state=0,min_samples_split=10,n_estimators=100)
    classifier.fit(X_train, Y_train)
   #print('Parameters of RF:', classifier.get_params())
    
    return classifier


def Predict(strategy):

    X,Y = LoadData()
    X_train, X_test, Y_train, Y_test = TrainTestSplit(X,Y)
    classifier = TrainRandomForest(X_train,Y_train)
    missing = np.isnan(X_test).any(axis=1)
    if strategy == 'abstain':
        
        Y_predict = classifier.predict(X_test[~missing])
        accuracy = len(Y_predict) / len(Y_test)
        missing_accuracy = 0
        #print('Number if items missing are =', np.sum(missing))
    
    if strategy == 'majorityClass':
        # Get the mode from train set, use the first value in the array
        mode_class = Y_train.mode().iloc[0]
        # Create and fill in the predict np array using that mode
        Y_predict = np.full(len(Y_test), mode_class)
        accuracy = accuracy_score(Y_test, Y_predict)
        missing_accuracy = 99
    
    if strategy == 'omitFeatures':
        # with omitFeatures set to true, getting the split accordingly
        X_train_omit, X_test_omit, Y_train_omit, Y_test_omit = TrainTestSplit(X,Y,omitFeatures=True)

        #Re-train the classifier
        classifier = TrainRandomForest(X_train_omit,Y_train_omit)
        Y_predict = classifier.predict(X_test_omit)

        # need to convert to numpy, and treat as type int due to misalignment issues due to dropping features
        Y_test_omit = Y_test_omit.to_numpy()
        Y_predict = Y_predict.astype(int)

        # accuracy
        missing_accuracy = accuracy_score(Y_test_omit[missing], Y_predict[missing])
        accuracy = accuracy_score(Y_test_omit, Y_predict)
    
    if strategy == 'imputeAvg':
       
        imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
        imputer.fit(X_train)
        imputed_data = imputer.transform(X_test)
        
        Y_predict = classifier.predict(imputed_data)

        missing_accuracy = accuracy_score(Y_test[missing], Y_predict[missing])
        accuracy = accuracy_score(Y_test,Y_predict)

    if strategy == 'imputeNeighbor':

        imputer = KNNImputer(n_neighbors=5)   
        imputer.fit(X_train)
        imputed_data = imputer.transform(X_test)
        
        Y_predict = classifier.predict(imputed_data)

        missing_accuracy = accuracy_score(Y_test[missing], Y_predict[missing])
        accuracy = accuracy_score(Y_test,Y_predict)


    return accuracy, missing_accuracy

if __name__ == "__main__":

    
    strategies = ['abstain','majorityClass', 'omitFeatures','imputeAvg','imputeNeighbor']
    strategyAccuracy = []
    for strategy in strategies: 
        accuracy,missing_accuracy = Predict(strategy)
        strategyAccuracy.append((strategy,accuracy,missing_accuracy))
    print(strategyAccuracy)