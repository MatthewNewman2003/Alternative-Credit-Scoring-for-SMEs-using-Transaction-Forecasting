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

#Selecting transaction dates for parsing
TransactionDates = TransactionData["TransactionDate"]

#Reformatting transaction dates as months
ParsedTransactionDates = [parser.parse(date) for date in TransactionDates]

FormattedTransactionMonths = [date.strftime("%m/%Y") for date in ParsedTransactionDates]

TransactionData["TransactionDate"] = FormattedTransactionMonths

#Checking the number of unique accounts
UniqueTransactions = TransactionData['AccountId'].drop_duplicates().reset_index(drop=True)

print(len(UniqueTransactions))

#Generating the distribution of number of transactions per account
AccountTransactionAmounts = TransactionData["AccountId"].value_counts()
AccountDistribution = AccountTransactionAmounts.value_counts().sort_index()
print(AccountDistribution)

#Checking the distribution of number of months per account
NumberOfMonthsPerAccount = TransactionData.groupby(by=["AccountId"])["TransactionDate"].nunique().sort_values(ascending=False)
print(NumberOfMonthsPerAccount)
NumberOfMonthsPerAccount = NumberOfMonthsPerAccount.reset_index()

#Visualising the distribution of number of months per account
BusinessMonthAmounts = NumberOfMonthsPerAccount["TransactionDate"]
plt.boxplot(BusinessMonthAmounts)
plt.title("Distribution of Number of Months per Account")
plt.ylabel("Number of Months")
plt.show()

#Removing all accounts that have less than 3 months of data to ensure that all models receive an adequate training set
for i in range(0, len(NumberOfMonthsPerAccount)):
    if NumberOfMonthsPerAccount.iloc[i]["TransactionDate"] < 3:
        TransactionData = TransactionData[TransactionData["AccountId"] != NumberOfMonthsPerAccount.iloc[i]["AccountId"]]

#Checking the updated number of unique accounts
UpdatedUniqueTransactions = TransactionData['AccountId'].drop_duplicates().reset_index(drop=True)

print(len(UpdatedUniqueTransactions))

#Checking the updated transaction amount distribution
AccountTransactionAmounts = TransactionData["AccountId"].value_counts()
AccountDistribution = AccountTransactionAmounts.value_counts().sort_index()
print(AccountDistribution)

#Detecting metadata from existing transaction dataset
TransactionsMetadata = Metadata.detect_from_dataframe(data=TransactionData, table_name="Transactions")
TransactionsMetadata.update_column(column_name='AccountId', sdtype='categorical')

#Fitting the synthetic data generator to the metadata from the original dataset
SyntheticDataGenerator = GaussianCopulaSynthesizer(TransactionsMetadata)
SyntheticDataGenerator.fit(TransactionData)

#Declaring the number of new accounts to be created and the number of transactions needed to parallel the original dataset
NumberOfNewAccounts = 800
np.random.seed(271)
TransactionCounts = np.random.choice(
    AccountDistribution.index,
    size=NumberOfNewAccounts,
    p=AccountDistribution.values/AccountDistribution.sum())

#Calculating the required number of rows
TotalRequiredRows = TransactionCounts.sum()
print(TotalRequiredRows)

#Generating synthetic data
Sample = SyntheticDataGenerator.sample(num_rows=int(TotalRequiredRows))

#Assigning account IDs to transactions in the synthetic data
NewAccountIDs = [str(uuid.uuid4()) for _ in range(NumberOfNewAccounts)]

AccountIDsRepeated = [AccountID for AccountID, Count in zip(NewAccountIDs, TransactionCounts) for _ in range(Count)]

AccountIDsRepeated = AccountIDsRepeated[:len(Sample)]
Sample['AccountId'] = AccountIDsRepeated

#Ensuring that no transaction misses an account ID and summarising the amount of new data generated
assert Sample['AccountId'].isna().sum() == 0
print(f"Total synthetic transactions: {len(Sample)}")
print(f"Unique synthetic accounts: {Sample['AccountId'].nunique()}")

#Combining the synthetic data with the original data and calculating the new amount of unique accounts
BoostedTransactions = pd.concat([TransactionData, Sample])
UniqueBoostedTransactions = BoostedTransactions['AccountId'].drop_duplicates().reset_index(drop=True)
print(len(UniqueBoostedTransactions))

#Checking the distribution of transaction amounts per account in the new combined dataset
print(BoostedTransactions)
BoostedTransactionAmounts = BoostedTransactions["AccountId"].value_counts()
BoostedAccountDistribution = BoostedTransactionAmounts.value_counts().sort_index()

#Converting any transactions labelled "Debit" into negative transactions, as Debit denotes money leaving the account
BoostedTransactions.loc[
    BoostedTransactions["TransactionType"].str.strip() == "Debit",
    "Amount"
] *= -1

#Ensuring that all transaction dates in the boosted dataset are parsed as months
TransactionDates = BoostedTransactions["TransactionDate"]

ParsedTransactionDates = [parser.parse(date) for date in TransactionDates]

FormattedTransactionMonths = [date.strftime("%m/%Y") for date in ParsedTransactionDates]

BoostedTransactions["TransactionDate"] = FormattedTransactionMonths

BoostedTransactions["TransactionDate"] = pd.to_datetime(BoostedTransactions["TransactionDate"])
BoostedTransactions["TransactionDate"] = BoostedTransactions["TransactionDate"].dt.to_period("M").dt.to_timestamp()

#Checking grouped number of transactions by account and month as a test
GroupedNumberOfTransactionsByAccountAndMonth = BoostedTransactions.groupby(by=["AccountId", "TransactionDate"]).count()

GroupedNumberOfTransactionsByAccountAndMonth = GroupedNumberOfTransactionsByAccountAndMonth.sort_values(by=["AccountId", "TransactionDate"])

#Checking that the distributions parallel each other using cosine similarity and showing the length of the boosted dataset
Similarity = cosine_similarity([AccountDistribution], [BoostedAccountDistribution])[0][0]
Correlation = np.corrcoef(AccountDistribution, BoostedAccountDistribution)[0, 1]
print(Similarity)
print(Correlation)
print(len(BoostedTransactions))

#Checking the number of months per account distribution in the boosted dataset
NumberOfMonthsPerAccount = BoostedTransactions.groupby(by=["AccountId"])["TransactionDate"].nunique().sort_values(ascending=False)
print(NumberOfMonthsPerAccount)
NumberOfMonthsPerAccount = NumberOfMonthsPerAccount.reset_index()

#Removing any synthetically generated accounts with less than 3 months of data to ensure that every model has sufficient training data
for i in range(0, len(NumberOfMonthsPerAccount)):
    if NumberOfMonthsPerAccount.iloc[i]["TransactionDate"] < 3:
        BoostedTransactions = BoostedTransactions[BoostedTransactions["AccountId"] != NumberOfMonthsPerAccount.iloc[i]["AccountId"]]

#Rechecking the number of months per account distribution after this
NumberOfMonthsPerAccount = BoostedTransactions.groupby(by=["AccountId"])["TransactionDate"].nunique().sort_values(ascending=False)
print(NumberOfMonthsPerAccount)

#Showing the final dataset length and number of unique accounts
print(len(BoostedTransactions))
print(BoostedTransactions["AccountId"].nunique())

#Using cosine similarity and correlation to check that the final synthetic and original distributions parallel each other
BoostedTransactionAmounts = BoostedTransactions["AccountId"].value_counts()
BoostedAccountDistribution = BoostedTransactionAmounts.value_counts().sort_index()
Similarity = cosine_similarity([AccountDistribution], [BoostedAccountDistribution])[0][0]
Correlation = np.corrcoef(AccountDistribution, BoostedAccountDistribution)[0, 1]
print(Similarity)
print(Correlation)

#Writing the boosted dataset to a CSV
BoostedTransactions.to_csv("Boosted Transaction Dataset.csv")
