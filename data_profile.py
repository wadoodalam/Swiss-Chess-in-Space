import pandas as pd
import numpy as np

# missing values are represented by 99/-99 in the dataset
data = pd.read_csv('cfhtlens.csv', na_values=[99,-99])



description = data.describe()
description = description.loc[['min', 'max', 'mean', '50%','count']]
description.to_csv('profile.csv')