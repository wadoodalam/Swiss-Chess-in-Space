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
    #print("Training set count:", len(Y_train))

    # Concatenate the missing value rows in the test set
    X_test = pd.concat([X_test, X[missing]])
    Y_test = pd.concat([Y_test, Y[missing]])

    return X_train, X_test, Y_train, Y_test

def TrainRandomForest(X_train, Y_train):
    classifier = RandomForestClassifier(random_state=0)
    classifier.fit(X_train, Y_train)
    return classifier


def Predict(classifier, X_test, Y_test, strategy):
    if strategy == 'abstain':
        missing = np.isnan(X_test).any(axis=1)
        Y_predict = classifier.predict(X_test[~missing])
        accuracy = len(Y_predict) / len(Y_test)
        print('Number if items missing are =', np.sum(missing))
    
    if strategy == 'majorityClass':
        # Get the mode from test set, use the first value in the array
        mode_class = Y_train.mode().iloc[0]
        # Create and fill in the predict np array using that mode
        Y_predict = np.full(len(Y_test), mode_class)
        accuracy = accuracy_score(Y_test, Y_predict)
    
    if strategy == 'omitFeatures':
        # Loading data again as we need to omit features
        X,Y = LoadData()
        # with omitFeatures set to true, getting the split accordingly
        X_train_omit, X_test_omit, Y_train_omit, Y_test_omit = TrainTestSplit(X,Y,omitFeatures=True)
        #Re-train the classifier
        classifier = TrainRandomForest(X_train_omit,Y_train_omit)
        # Prediction and accuracy
        Y_predict = classifier.predict(X_test_omit)
        accuracy = accuracy_score(Y_test_omit, Y_predict)
    
    if strategy == 'imputeAvg':
        X,Y = LoadData()
        X_train, X_test, Y_train, Y_test = TrainTestSplit(X,Y)
       
        imputer = SimpleImputer(missing_values=np.nan,strategy='mean')
        imputer.fit(X_train)
        imputed_data = imputer.transform(X_test)
        
        Y_predict = classifier.predict(imputed_data)
        accuracy = accuracy_score(Y_test,Y_predict)

    if strategy == 'imputeNeighbor':
        X,Y = LoadData()
        X_train, X_test, Y_train, Y_test = TrainTestSplit(X,Y)

        imputer = KNNImputer()   
        imputer.fit(X_train)
        imputed_data = imputer.transform(X_test)
        
        Y_predict = classifier.predict(imputed_data)
        accuracy = accuracy_score(Y_test,Y_predict)


    return accuracy

if __name__ == "__main__":

    X,Y = LoadData()
    X_train, X_test, Y_train, Y_test = TrainTestSplit(X,Y)
    classifier = TrainRandomForest(X_train,Y_train)
    
    accuracy = Predict(classifier, X_test, Y_test, 'imputeNeighbor')
    print(f'Accuracy: {accuracy:.4f}')