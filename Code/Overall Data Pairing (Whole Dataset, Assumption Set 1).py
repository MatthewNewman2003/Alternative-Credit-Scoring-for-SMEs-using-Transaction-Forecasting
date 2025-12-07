#Importing libraries
import pandas as pd
import random
from datetime import datetime
import datetime as dt
from dateutil import parser
from scipy.spatial import KDTree
import numpy as np

#Reading in the datasets and filtering the loan data to only include repayers and defaulters
BusinessAccountData = pd.read_csv("../Menna Dissertation Dataset/MSc project/open_src_data/synthetic_uk_business_accounts/synthetic_uk_business_current_accounts.csv")
print(BusinessAccountData)
TransactionData = pd.read_csv("Boosted Transaction Dataset.csv")
LoanData = pd.read_csv("../Menna Dissertation Dataset/MSc project/open_src_data/sme_loans/sme_loans_data.csv")
print(LoanData)
RepaidLoanData = LoanData[LoanData["status"] == "Paid"]
DefaultedLoanData = LoanData[LoanData["status"] == "Defaulted"]
LoanData = pd.concat([RepaidLoanData, DefaultedLoanData], axis=0, ignore_index=True)
print(LoanData)
print(DefaultedLoanData["status"])

#Calculating the number of unique business accounts and transaction accounts and resultantly calculating the maximum number of pairs
UniqueBusinesses = BusinessAccountData['current_account_number'].drop_duplicates().reset_index(drop=True)
UniqueTransactions = TransactionData['AccountId'].drop_duplicates().reset_index(drop=True)

MaximumPairs = min(len(UniqueBusinesses), len(UniqueTransactions))

#Converting Debit transaction amounts into negative values
for i in range(0, len(TransactionData)):
    if TransactionData.iloc[i]["TransactionType"] == "Debit":
        TransactionData.at[i, "Amount"] = -(TransactionData.at[i, "Amount"])

#Converting transaction dates into months
TransactionDates = TransactionData["TransactionDate"]

ParsedTransactionDates = [parser.parse(date) for date in TransactionDates]

FormattedTransactionMonths = [date.strftime("%m/%Y") for date in ParsedTransactionDates]

TransactionData["TransactionDate"] = FormattedTransactionMonths

#Grouping transactions by account/month and account and calculating the net transaction amounts for each grouping
GroupedSumOfTransactionsByAccountAndMonth = TransactionData.groupby(by=["AccountId", "TransactionDate"]).agg(["sum"])["Amount"]
print(GroupedSumOfTransactionsByAccountAndMonth)

GroupedSumOfTransactionsByAccount = TransactionData.groupby(by="AccountId").agg(["sum"])["Amount"]
print(GroupedSumOfTransactionsByAccount)

GroupedSumOfTransactionsByAccount = GroupedSumOfTransactionsByAccount.reset_index(drop=False)
print(GroupedSumOfTransactionsByAccount)

#Calculating the cash flow (revenue-costs) for each business account
BusinessAccountData["Cash Flow"] = BusinessAccountData["2019_revenue"] - BusinessAccountData["costs"]

#Calculating cash flow and transaction amount percentiles for each business account and transaction account respectively
BusinessAccountData["Cash Flow Percentile"] = BusinessAccountData["Cash Flow"].rank(pct=True)
GroupedSumOfTransactionsByAccount["Transaction Amount Percentile"] = GroupedSumOfTransactionsByAccount["sum"].rank(pct=True)

print(BusinessAccountData['Cash Flow Percentile'].isnull().sum())
print(GroupedSumOfTransactionsByAccount['Transaction Amount Percentile'].isnull().sum())

BusinessAccountData['Cash Flow'] = BusinessAccountData['Cash Flow'].fillna(0)
BusinessAccountData["Cash Flow Percentile"] = BusinessAccountData["Cash Flow"].rank(pct=True)

#Pairing business accounts with transaction accounts using percentile-based pairing
Tree = KDTree(GroupedSumOfTransactionsByAccount[["Transaction Amount Percentile"]].values)

Distances, Indices = Tree.query(BusinessAccountData[["Cash Flow Percentile"]].values, k=1)

used_indices = set()
final_matches = []

for i, idx in enumerate(Indices.flatten()):
    if idx not in used_indices:
        final_matches.append((i, idx))
        used_indices.add(idx)

print(final_matches)

#Generating mappings of business accounts to transaction accounts based on this pairing
MatchedDataFrame = pd.DataFrame({
    'Company Registration Number': BusinessAccountData.iloc[[i for i, _ in final_matches]]['company_reg_number'].values,
    'Current Account Number': BusinessAccountData.iloc[[i for i, _ in final_matches]]['current_account_number'].values,
    'Cash Flow': BusinessAccountData.iloc[[i for i, _ in final_matches]]['Cash Flow'].values,
    'Primary Sector': BusinessAccountData.iloc[[i for i, _ in final_matches]]["primary_sector"].values,
    'Matched AccountID': GroupedSumOfTransactionsByAccount.iloc[[j for _, j in final_matches]]['AccountId'].values,
    'Transaction Sum': GroupedSumOfTransactionsByAccount.iloc[[j for _, j in final_matches]]['sum'].values,
})

print(MatchedDataFrame)

NumberToIDMappings = {}
for i in range(0, len(MatchedDataFrame)):
    AccountNumber = MatchedDataFrame.iloc[i]["Current Account Number"]
    AccountID = MatchedDataFrame.iloc[i]["Matched AccountID"]
    NumberToIDMappings[AccountNumber] = AccountID

BusinessAccountData["AccountID"] = BusinessAccountData["current_account_number"].map(NumberToIDMappings)
BusinessAccountData = BusinessAccountData.dropna(subset=["AccountID"])

#Reading in cumulative and month-by-month forecast data
CumulativeForecastData = pd.read_csv("Naive Forecasts (Cumulative).csv")
MonthByMonthForecastData = pd.read_csv("SSA Forecasts (Month-by-Month).csv")

#Making account IDs interpretable and merging forecasts with businesses
BusinessAccountData["AccountID"] = BusinessAccountData["AccountID"].astype("str")
BusinessAccountData["AccountID"] = BusinessAccountData["AccountID"].astype("object")
BusinessAccountData["AccountID"] = BusinessAccountData["AccountID"].astype(str).str.strip()
CumulativeForecastData["AccountID"] = CumulativeForecastData["AccountID"].astype(str).str.strip()
MonthByMonthForecastData["AccountID"] = MonthByMonthForecastData["AccountID"].astype(str).str.strip()

MergedBusinessesAndForecasts = pd.merge(BusinessAccountData, CumulativeForecastData, on="AccountID", how="inner")
MergedBusinessesAndForecasts = pd.merge(MergedBusinessesAndForecasts, MonthByMonthForecastData, on="AccountID", how="inner")
print(MergedBusinessesAndForecasts)

print(LoanData)

#Setting a random seed for reproducibility
np.random.seed(43)
Today = datetime.today()

#Randomly pairing business accounts to loan accounts
BusinessToLoanMatches = []

MaximumPairs = min(len(MergedBusinessesAndForecasts), len(LoanData))

SampledBusinesses = MergedBusinessesAndForecasts["AccountID"].sample(n=MaximumPairs, replace=False).reset_index(drop=True)
SampledLoans = LoanData["accountId"].sample(n=MaximumPairs, replace=False).reset_index(drop=True)

Matches = list(zip(SampledBusinesses, SampledLoans))
for i in Matches:
    BusinessToLoanMatches.append({
        "Business Account ID" : i[0],
        "Loan Account ID" : i[1]
    })

BusinessAndLoanPairings = pd.DataFrame(BusinessToLoanMatches)

#Merging the business account and loan account datasets based on the generated random pairings
LoanToBusinessMappings = {}
for i in range(0, len(BusinessAndLoanPairings)):
    BusinessAccountID = BusinessAndLoanPairings.iloc[i]["Business Account ID"]
    LoanAccountID = BusinessAndLoanPairings.iloc[i]["Loan Account ID"]
    LoanToBusinessMappings[LoanAccountID] = BusinessAccountID

LoanData["AccountID"] = LoanData["accountId"].map(LoanToBusinessMappings)
LoanData = LoanData.dropna(subset=["AccountID"])
MergedDataset = pd.merge(MergedBusinessesAndForecasts, LoanData, on="AccountID", how="inner")

#Selecting only required columns from the merged dataset and writing it into a CSV
MergedDataset = MergedDataset[["AccountID", "Starting Amount", "Ending Amount", "Net Change over Forecast Period", "Highest Amount", "Lowest Amount",
                               "Difference", "primary_sector", "company_type", "2019_revenue", "costs", "accounts_receivable",
                               "capital_and_reserves", "current_assets", "current_liabilities", "fixed_assets", "long_term_liabilities",
                               "provisions_for_liabilities", "loanAmount", "yearsOfCreditHistory", "totalCreditLines", "openCreditLines", "bankruptcies",
                               "delinquencies", "netValue", "status"]]
MergedDataset = MergedDataset.dropna()
MergedDataset.to_csv("Merged Dataset (Whole Dataset, Assumption Set 1).csv")
