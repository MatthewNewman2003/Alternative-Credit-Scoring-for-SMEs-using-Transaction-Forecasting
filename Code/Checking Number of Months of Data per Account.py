#Importing libraries
import uuid
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil import parser
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#Reading in the data and selecting relevant columns
TransactionData = pd.read_csv("synthetic_ob_transactions.csv")
TransactionData = TransactionData[["TransactionDate", "TransactionType", "Amount", "AccountId"]]
print(TransactionData)

#Converting transaction dates into months
TransactionDates = TransactionData["TransactionDate"]

ParsedTransactionDates = [parser.parse(date) for date in TransactionDates]

FormattedTransactionMonths = [date.strftime("%m/%Y") for date in ParsedTransactionDates]

TransactionData["TransactionDate"] = FormattedTransactionMonths

#Checking the number of months per account
NumberOfMonthsPerAccount = TransactionData.groupby(by=["AccountId"])["TransactionDate"].nunique().sort_values(ascending=False)
print(NumberOfMonthsPerAccount)
NumberOfMonthsPerAccount = NumberOfMonthsPerAccount.reset_index()

#Visualising the distribution of number of months per account
BusinessMonthAmounts = NumberOfMonthsPerAccount["TransactionDate"]
plt.boxplot(BusinessMonthAmounts)
plt.title("Distribution of Number of Months per Account")
plt.ylabel("Number of Months")
plt.show()
