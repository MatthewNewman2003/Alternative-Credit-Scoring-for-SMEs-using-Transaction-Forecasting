#Importing libraries
import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

#Declaring a random seed for reproducibility
Seed = 43

#Reading in the data
Data = pd.read_csv("Merged Dataset (Whole Dataset, Assumption Set 3).csv")

#Declaring X and Y
X = Data[["primary_sector", "company_type", "2019_revenue", "costs", "accounts_receivable", "capital_and_reserves", "current_assets", "current_liabilities",
          "fixed_assets", "long_term_liabilities", "provisions_for_liabilities", "loanAmount", "yearsOfCreditHistory", "totalCreditLines", "openCreditLines",
          "bankruptcies", "delinquencies", "netValue", "Highest Amount", "Lowest Amount", "Difference"]]
Y = Data["status"]

#One-hot encoding categorical X columns
XNumerical = X.drop(columns=["primary_sector", "company_type"])
XCategorical = X[["primary_sector", "company_type"]]

OneHotEncoder = OneHotEncoder(sparse_output=False)
XCategoricalEncoded = pd.DataFrame(OneHotEncoder.fit_transform(XCategorical), columns=OneHotEncoder.get_feature_names_out(XCategorical.columns))

XEncoded = pd.concat([XNumerical.reset_index(drop=True), XCategoricalEncoded.reset_index(drop=True)], axis=1)

#Label encoding Y
LblEncoder = LabelEncoder()
YEncoded = LblEncoder.fit_transform(Y)
print(LblEncoder.classes_)

#Splitting the data into training and testing sets
XTrain, XTest, YTrain, YTest = train_test_split(XEncoded, YEncoded, test_size=0.3, stratify=YEncoded, random_state=Seed)

#Determining the positive class weighting
NumNegative = sum(YTrain == 0)
NumPositive = sum(YTrain == 1)
PositiveWeighting = NumNegative/NumPositive

#Declaring the parameter grid for the grid search
Parameters = {"n_estimators" : [25, 50, 100, 200],
              "eta" : [0.01, 0.05, 0.1, 0.2],
              "max_depth" : [1, 2, 4, 8]}

#Performing a grid search to find the best XGBoost parameters
print("Train Length After Resampling:",len(YTrain))
print("Test Length:",len(YTest))
Model = xgb.XGBClassifier(n_estimators=100, objective='binary:logistic', tree_method='hist', scale_pos_weight=PositiveWeighting, random_state=Seed)
Validator = StratifiedKFold(n_splits=10, shuffle=True, random_state=Seed)
Search = GridSearchCV(Model, Parameters, scoring="roc_auc", n_jobs=1, cv=Validator)
Results = Search.fit(XTrain, YTrain)

#Taking the best model, making predictions with it and outputting the results
BestModel = Results.best_estimator_

RegularPredictions = BestModel.predict(XTest)
Accuracy = accuracy_score(RegularPredictions, YTest)
Probabilities = BestModel.predict_proba(XTest)[:,1]
AUC = roc_auc_score(YTest, Probabilities)
PR_AUC = average_precision_score(YTest, Probabilities, pos_label=0)
Metrics = classification_report(YTest, RegularPredictions, target_names=["Defaulted", "Repaid"])
print(Metrics)
ConfusionMatrix = confusion_matrix(YTest, RegularPredictions)
print(ConfusionMatrix)
