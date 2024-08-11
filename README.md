<h1> 
  Prediction: Breast Cancer via Random Forest
</h1>

## Business Problem

In this section, we plan to predict whether breast cancer is **malignant** or **benign** by specific parameters.

Dataset Story
Breast cancer is the most common cancer among women in the world. It accounts for 25% of all cancer cases and affected over 2.1 Million people in 2015 alone. It starts when cells in the breast begin to grow out of control. These cells usually form tumors that can be seen via X-ray or felt as lumps in the breast area.

The key challenge against its detection is how to classify tumors into malignant (cancerous) or benign(non-cancerous). We ask you to complete the analysis of classifying these tumors using machine learning (with SVMs) and the Breast Cancer Wisconsin (Diagnostic) Dataset.

## Necessary Libraries

Required libraries and some settings for this section are:

```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.float_format", lambda x: "%.4f" % x)

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

import warnings
warnings.filterwarnings("ignore")
```

## Importing the Dataset

First, we import the dataset `breast-cancer.csv` into the pandas DataFrame.

## General Information About the Dataset

### Checking the Data Frame

As we want to check the data to have a general opinion about it, we create and use a function called `check_df(dataframe, head=5, tail=5)` that prints the referred functions:


    dataframe.head(head)
    
    dataframe.tail(tail)
    
    dataframe.shape
    
    dataframe.dtypes
    
    dataframe.size
    
    dataframe.isnull().sum()
    
    dataframe.describe([0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1]).T

### Removing Irrelevant Features

    df = df.drop(columns=['id'], axis=1)

By this expression, we eliminate `id` feature since it is irrelevant to us.

## Analysis of Categorical and Numerical Variables

After checking the data frame, we need to define and separate columns as **categorical** and **numerical**. We define a function called `grab_col_names` for separation that benefits from multiple list comprehensions as follows:

    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ['category', 'object', 'bool']]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes in ['uint8', 'int64', 'int32', 'float64']]
    cat_but_car = [col for col in df.columns if df[col].nunique() > car_th and str(df[col].dtypes) in ['object', 'category']]
    cat_cols = cat_cols + num_but_cat
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ['uint8', 'int64', 'float64']]
    num_cols = [col for col in num_cols if col not in cat_cols]

`cat_th` and `car_th` are the threshold parameters to decide the column type.

**Categorical Columns:**

* diagnosis

**Numerical Columns:**

Rest of the features are numerical

### Summarization and Visualization of the Categorical and Numerical Columns

To summarize and visualize the referred column we create two other functions called `cat_summary` and `num_summary`.

For example, categorical column **diagnosis**:

############### diagnosis ###############

diagnosis | diagnosis count | Ratio |
--------|------------|-------|
0       |    500  |  65.1042 |
1       |    268  |  34.8958 |

![__results___16_1](https://github.com/user-attachments/assets/8733d914-ba37-4b3f-8455-969f71fc3e00)

Another example is, the numerical column **radius_mean**:

############### radius_mean ###############

Process | Result |
--------|--------|
count |  569.0000 |
mean   |  14.1273 |
std    |   3.5240 |
min    |   6.9810 |
1%     |   8.4584 |
5%     |   9.5292 |
10%    |  10.2600 |
20%    |  11.3660 |
30%    |  12.0120 |
40%    |  12.7260 |
50%    |  13.3700 |
60%    |  14.0580 |
70%   |   15.0560 |
80%   |   17.0680 |
90%   |   19.5300 |
95%   |   20.5760 |
99%   |   24.3716 |
max   |   28.1100 |

Name: radius_mean, dtype: float64

![__results___19_1](https://github.com/user-attachments/assets/4cfb18e3-2a48-4a12-8784-799453157dfb)

With the help of a for loop we apply these functions to all columns in the data frame.

We create another plot function called `plot_num_summary(dataframe)` to see the whole summary of numerical columns due to the high quantity of them:

![__results___21_0](https://github.com/user-attachments/assets/6b04377c-4464-4031-93a9-eafbb4b37b6b)

## Target Analysis

We create another function called `target_summary_with_cat(dataframe, target, categorical_col)` to examine the target by categorical features.

For instance *radius_mean*

################ diagnosis --> radius_mean #################

diagnosis | radius_mean |
--------|---------|
0   |     12.1465 |
1   |     17.4628 |

## Correlation Analysis

To analyze correlations between numerical columns we create a function called `correlated_cols(dataframe)`:

![__results___28_0](https://github.com/user-attachments/assets/b77835ab-82d6-4d97-90f8-c8d0d69718b7)

### High Correlated Features

We analyze the correlations once more with a threshold of 90%:

High correlated features:

* perimeter_mean
* area_mean
* concave points_mean
* perimeter_se
* area_se
* radius_worst
* texture_worst
* perimeter_worst
* area_worst
* concave points_worst

## Missing Value Analysis

We check the data to designate the missing values in it, `dataframe.isnull().sum()`:

* diagnosis                  0
* radius_mean                0
* texture_mean               0
* perimeter_mean             0
* area_mean                  0
* smoothness_mean            0
* compactness_mean           0
* concavity_mean             0
* concave points_mean        0
* symmetry_mean              0
* fractal_dimension_mean     0
* radius_se                  0
* texture_se                 0
* perimeter_se               0
* area_se                    0
* smoothness_se              0
* compactness_se             0
* concavity_se               0
* concave points_se          0
* symmetry_se                0
* fractal_dimension_se       0
* radius_worst               0
* texture_worst              0
* perimeter_worst            0
* area_worst                 0
* smoothness_worst           0
* compactness_worst          0
* concavity_worst            0
* concave points_worst       0
* symmetry_worst             0
* fractal_dimension_worst    0

dtype: int64

## Random Forest: Machine Learning Algorithm

At last, we create our model and see the results:

******************** Accuracy & Results ********************

Accuracy Train:  1.000

Accuracy Test:  0.971

R2 Train:  1.000

R2 Test:  0.971

Cross Validation Train: 0.950

Cross Validation Test: 0.959

Cross Validation (Accuracy): 0.956

Cross Validation (F1):  0.940

Cross Validation (Precision): 0.953

Cross Validation (Recall): 0.930

Cross Validation (ROC Auc): 0.991

![__results___37_1](https://github.com/user-attachments/assets/9aef55ac-f309-4ba1-8a9d-0c471e493854)

## Loading a Base Model and Prediction

Via **joblib** we can save and/or load our model:

    def load_model(pklfile):
      model_disc = joblib.load(pklfile)
      return model_disc
      
    model_disc = load_model("rf_model.pkl")

________

Now we can make predictions with our model:
    
    X = df.drop("diagnosis", axis=1)
    x = X.sample(1).values.tolist()

    model_disc.predict(pd.DataFrame(X))[0]
    1

________

    sample2 = [13, 8, 125, 1000, 0.12, 0.25, 0.4, 0.2, 0.16, 0.07, 1, 0.8, 9, 153, 0.005, 0.05, 0.06, 0.02, 0.02, 0.005, 26, 18, 185, 2020, 0.17, 0.5, 0.9, 0.28, 0.44, 0.12]

    model_disc.predict(pd.DataFrame(sample2).T)[0]
    1
