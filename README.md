## InfoImputer

In the field of data science, one of the most common challenges is preparing datasets for machine learning algorithms. Dealing with missing values in the dataset is a critical aspect of this process. To address this challenge, data scientists have developed various imputation techniques that aim to accurately fill these missing values.

Among the popular imputers are:

SimpleImputer: This imputer fills missing values in the data using statistical properties such as mean, median, or the most frequent value.

KNNImputer: The KNNImputer completes missing values by utilizing the k-nearest neighbors algorithm.

IterativeImputer: This imputer estimates each feature from all the other features in an iterative manner.

## Introducing InfoImputer:

It is seems that it performs better than the aforementioned imputers.
It is similar in nature to the IterativeImputer but comes with some notable differences:

Handling uncorrelated features: The IterativeImputer uses a hyperparameter called n_nearest_features, which determines the number of other features used to estimate missing values for each feature column. However, using all other columns to estimate the target feature may lead to weak predictions and slower processing, especially when the features are uncorrelated. In contrast, InfoImputer has two different approaches:
one sets an absolute correlation coefficient threshold to select only the most relevant features for estimatio and the other consider the n most informative features for the specific feature. These ensure a more effective and efficient imputation process.

Separate estimators for classification and regression: The IterativeImputer uses a single estimator for both categorical and numerical columns. However, InfoImputer recognizes the different nature of classification and regression tasks and employs separate estimators for each type. This tailored approach leads to more accurate imputed values.

Automated conversion of categorical values: In the IterativeImputer, converting categorical values to numeric format needs to be done manually. InfoImputer automates this process by factorizing categorical values into numeric representations. This simplifies the imputation workflow, particularly when dealing with categorical data.

By addressing these issues, InfoImputer offers an improved approach to handling missing values in datasets. It takes into account the correlation and mutual information score between features, utilizes separate estimators for classification and regression tasks, and automates the conversion of categorical values to numeric representations.

## The Main Motivation

As I showed in the following notebook, correlation can only find linear dependency between random variables where mutual information can also detect nonlinear relations and dependencies.

https://www.kaggle.com/code/khashayarrahimi94/why-you-should-not-use-correlation

Therefore, besides some automation and ease of use in this imputer, I add mutual information score as a criteria for selecting dependece and informational features for the features with missing values that we want to fill them.

## Install

```shell
pip install AutoImputer
```
## Example

```python
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingRegressor
import InfoImputer
from InfoImputer.Auto import Imputer
#import the data in pandas format
data = pd.read_csv(r"your directory")

#if you want to use the correlation coefficient threshold (here threshold = 0.1):
FilledData = Imputer(data,TargetName,0.1,GradientBoostingRegressor,ExtraTreesClassifier)

#if you want to use N most informative features using mutual information (here N = 3)
FilledData = Imputer(data,TargetName,3,GradientBoostingRegressor,ExtraTreesClassifier)
```



