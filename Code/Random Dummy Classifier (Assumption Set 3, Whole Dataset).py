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
from sklearn.dummy import DummyClassifier

#Setting random seed for reproducibility
Seed = 43

#Reading in the data
Data = pd.read_csv("Merged Dataset (Whole Dataset, Assumption Set 3).csv")

#Declaring X and Y
X = Data[["primary_sector", "company_type", "2019_revenue", "costs", "accounts_receivable", "capital_and_reserves", "current_assets", "current_liabilities",
          "fixed_assets", "long_term_liabilities", "provisions_for_liabilities", "loanAmount", "yearsOfCreditHistory", "totalCreditLines", "openCreditLines",
          "bankruptcies", "delinquencies", "netValue", "Highest Amount", "Lowest Amount", "Difference", "Starting Amount", "Ending Amount",
          "Net Change over Forecast Period"]]
Y = Data["status"]

#One-hot encoding categorical X variables
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

#Making predictions with the dummy classifier and outputting the results
Classifier = DummyClassifier(strategy="uniform")
Classifier.fit(XTrain, YTrain)

RegularPredictions = Classifier.predict(XTest)
Metrics = classification_report(YTest, RegularPredictions, target_names=["Defaulted", "Repaid"])
print(Metrics)
ConfusionMatrix = confusion_matrix(YTest, RegularPredictions)
print(ConfusionMatrix)
