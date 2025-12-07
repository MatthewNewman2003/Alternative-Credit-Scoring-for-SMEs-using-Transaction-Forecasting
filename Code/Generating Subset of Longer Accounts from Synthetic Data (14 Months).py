#Importing libraries
import pandas as pd
from datetime import datetime
from dateutil import parser

#Reading in the data
TransactionData = pd.read_csv("Boosted Transaction Dataset.csv")
print(TransactionData.head())

#Extracting transaction dates and converting them into months
TransactionDates = TransactionData["TransactionDate"]

ParsedTransactionDates = [parser.parse(date) for date in TransactionDates]

FormattedTransactionMonths = [date.strftime("%m/%Y") for date in ParsedTransactionDates]

TransactionData["TransactionDate"] = FormattedTransactionMonths

print(TransactionData.head())

#Checking the number of months per account
NumberOfMonthsPerAccount = TransactionData.groupby(by=["AccountId"])["TransactionDate"].nunique().sort_values(ascending=False)
NumberOfMonthsPerAccount = NumberOfMonthsPerAccount.reset_index()
print(NumberOfMonthsPerAccount)
print(len(NumberOfMonthsPerAccount))

#Dneoting the longer subset by selecting only accounts with 14 or more months of data
LongerAccounts = NumberOfMonthsPerAccount[NumberOfMonthsPerAccount["TransactionDate"] >= 14]
print(LongerAccounts)
print(len(LongerAccounts))

#Generating a set of longer accounts and limiting the longer subset to only accounts that are in that set
LongerAccountsSet = set(LongerAccounts["AccountId"])

LongerSubset = TransactionData[TransactionData["AccountId"].isin(LongerAccountsSet)]

#Checking the lengths of the transaction dataset and the longer subset
print(len(TransactionData))
print(len(LongerSubset))

#Writing the longer subset to a CSV
LongerSubset.to_csv("Longer Subset of Transaction Data (14 Months).csv")
