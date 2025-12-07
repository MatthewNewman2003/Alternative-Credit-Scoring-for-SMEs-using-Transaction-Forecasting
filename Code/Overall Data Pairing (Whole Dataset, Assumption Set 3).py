#Importing libraries
import pandas as pd
import random
from datetime import datetime
import datetime as dt
from dateutil import parser
from scipy.spatial import KDTree
import numpy as np

#Reading in datasets and filtering loan data to only encompass repayers and defaulters
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

#Calculating the number of unique business accounts and unique transaction accounts
UniqueBusinesses = BusinessAccountData['current_account_number'].drop_duplicates().reset_index(drop=True)
UniqueTransactions = TransactionData['AccountId'].drop_duplicates().reset_index(drop=True)

#Determining the maximum number of business-transaction account pairings that can be made
MaximumPairs = min(len(UniqueBusinesses), len(UniqueTransactions))

#Converting all Debit transactions into negative values
for i in range(0, len(TransactionData)):
    if TransactionData.iloc[i]["TransactionType"] == "Debit":
        TransactionData.at[i, "Amount"] = -(TransactionData.at[i, "Amount"])

#Converting transaction dates into months
TransactionDates = TransactionData["TransactionDate"]

ParsedTransactionDates = [parser.parse(date) for date in TransactionDates]

FormattedTransactionMonths = [date.strftime("%m/%Y") for date in ParsedTransactionDates]

TransactionData["TransactionDate"] = FormattedTransactionMonths

#Grouping transactions by account/month and account and calculating the net amount for each grouping
GroupedSumOfTransactionsByAccountAndMonth = TransactionData.groupby(by=["AccountId", "TransactionDate"]).agg(["sum"])["Amount"]
print(GroupedSumOfTransactionsByAccountAndMonth)

GroupedSumOfTransactionsByAccount = TransactionData.groupby(by="AccountId").agg(["sum"])["Amount"]
print(GroupedSumOfTransactionsByAccount)

GroupedSumOfTransactionsByAccount = GroupedSumOfTransactionsByAccount.reset_index(drop=False)
print(GroupedSumOfTransactionsByAccount)

#Calculating each business account's cash flow (revenue-costs)
BusinessAccountData["Cash Flow"] = BusinessAccountData["2019_revenue"] - BusinessAccountData["costs"]

#Calculating each business account's cash flow percentile and each transaction amount's transaction amount percentile
BusinessAccountData["Cash Flow Percentile"] = BusinessAccountData["Cash Flow"].rank(pct=True)
GroupedSumOfTransactionsByAccount["Transaction Amount Percentile"] = GroupedSumOfTransactionsByAccount["sum"].rank(pct=True)

#Checking for no null percentile values
print(BusinessAccountData['Cash Flow Percentile'].isnull().sum())
print(GroupedSumOfTransactionsByAccount['Transaction Amount Percentile'].isnull().sum())

BusinessAccountData['Cash Flow'] = BusinessAccountData['Cash Flow'].fillna(0)
BusinessAccountData["Cash Flow Percentile"] = BusinessAccountData["Cash Flow"].rank(pct=True)

#Matching business accounts to transaction accounts by percentile-based pairing
Tree = KDTree(GroupedSumOfTransactionsByAccount[["Transaction Amount Percentile"]].values)

Distances, Indices = Tree.query(BusinessAccountData[["Cash Flow Percentile"]].values, k=1)

used_indices = set()
final_matches = []

for i, idx in enumerate(Indices.flatten()):
    if idx not in used_indices:
        final_matches.append((i, idx))
        used_indices.add(idx)

print(final_matches)

#Putting the matches into a DataFrame
MatchedDataFrame = pd.DataFrame({
    'Company Registration Number': BusinessAccountData.iloc[[i for i, _ in final_matches]]['company_reg_number'].values,
    'Current Account Number': BusinessAccountData.iloc[[i for i, _ in final_matches]]['current_account_number'].values,
    'Cash Flow': BusinessAccountData.iloc[[i for i, _ in final_matches]]['Cash Flow'].values,
    'Primary Sector': BusinessAccountData.iloc[[i for i, _ in final_matches]]["primary_sector"].values,
    'Matched AccountID': GroupedSumOfTransactionsByAccount.iloc[[j for _, j in final_matches]]['AccountId'].values,
    'Transaction Sum': GroupedSumOfTransactionsByAccount.iloc[[j for _, j in final_matches]]['sum'].values,
})

print(MatchedDataFrame)

#Mapping business accounts to transaction accounts
NumberToIDMappings = {}
for i in range(0, len(MatchedDataFrame)):
    AccountNumber = MatchedDataFrame.iloc[i]["Current Account Number"]
    AccountID = MatchedDataFrame.iloc[i]["Matched AccountID"]
    NumberToIDMappings[AccountNumber] = AccountID

BusinessAccountData["AccountID"] = BusinessAccountData["current_account_number"].map(NumberToIDMappings)
BusinessAccountData = BusinessAccountData.dropna(subset=["AccountID"])

#Reading in the cumulative and month-by-month forecast variables
CumulativeForecastData = pd.read_csv("Naive Forecasts (Cumulative).csv")
MonthByMonthForecastData = pd.read_csv("SSA Forecasts (Month-by-Month).csv")

#Ensuring that account IDs can be interpreted
BusinessAccountData["AccountID"] = BusinessAccountData["AccountID"].astype("str")
BusinessAccountData["AccountID"] = BusinessAccountData["AccountID"].astype("object")
BusinessAccountData["AccountID"] = BusinessAccountData["AccountID"].astype(str).str.strip()
CumulativeForecastData["AccountID"] = CumulativeForecastData["AccountID"].astype(str).str.strip()
MonthByMonthForecastData["AccountID"] = MonthByMonthForecastData["AccountID"].astype(str).str.strip()

#Joining businesses with their respective transaction forecasts and creating a merged DataFrame
MergedBusinessesAndForecasts = pd.merge(BusinessAccountData, CumulativeForecastData, on="AccountID", how="inner")
MergedBusinessesAndForecasts = pd.merge(MergedBusinessesAndForecasts, MonthByMonthForecastData, on="AccountID", how="inner")
print(MergedBusinessesAndForecasts)

print(LoanData)

#Setting a random seed for reproducibility
np.random.seed(43)
Today = datetime.today()

#Calculating cash flow percentiles in the merged dataset
MergedBusinessesAndForecasts["Cash Flow Percentile Rank"] = MergedBusinessesAndForecasts["Cash Flow"].rank(pct=True)

#Calculating loan amount percentiles in the loan dataset
LoanData["Loan Amount Percentile Rank"] = LoanData["loanAmount"].rank(pct=True)

MergedBusinessesAndForecasts = MergedBusinessesAndForecasts.sort_values("Cash Flow Percentile Rank").reset_index(drop=True)
LoanData = LoanData.sort_values("Loan Amount Percentile Rank").reset_index(drop=True)

#Pairing business accounts to loan accounts on the assumption that businesses with weaker cash flow request larger loans
UsedBusinesses = set()
BusinessToLoanMatches = []

for IndexLoan, RowLoan in LoanData.iterrows():
    UnusedBusinesses = MergedBusinessesAndForecasts[~MergedBusinessesAndForecasts["AccountID"].isin(UsedBusinesses)]
    
    Differences = (UnusedBusinesses["Cash Flow Percentile Rank"] - (1 - RowLoan["Loan Amount Percentile Rank"])).abs()
    MinimumDifference = Differences.min()
    
    CandidateIndices = Differences[Differences == MinimumDifference].index.tolist()
    ChosenIndex = np.random.choice(CandidateIndices)
    
    BusinessToLoanMatches.append({
        "Business Account ID" : MergedBusinessesAndForecasts.loc[ChosenIndex, "AccountID"],
        "Cash Flow" : MergedBusinessesAndForecasts.loc[ChosenIndex, "Cash Flow"],
        "Cash Flow Percentile Rank" : MergedBusinessesAndForecasts.loc[ChosenIndex, "Cash Flow Percentile Rank"],
        "Loan Account ID" : RowLoan["accountId"],
        "Loan Amount" : RowLoan["loanAmount"],
        "Loan Amount Percentile Rank" : RowLoan["Loan Amount Percentile Rank"]
    })
    
    UsedBusinesses.add(MergedBusinessesAndForecasts.loc[ChosenIndex, "AccountID"])

#Generating pairings of business accounts to loan accounts according to the percentile-based pairing just done
BusinessAndLoanPairings = pd.DataFrame(BusinessToLoanMatches)

LoanToBusinessMappings = {}
for i in range(0, len(BusinessAndLoanPairings)):
    BusinessAccountID = BusinessAndLoanPairings.iloc[i]["Business Account ID"]
    LoanAccountID = BusinessAndLoanPairings.iloc[i]["Loan Account ID"]
    LoanToBusinessMappings[LoanAccountID] = BusinessAccountID

LoanData["AccountID"] = LoanData["accountId"].map(LoanToBusinessMappings)
LoanData = LoanData.dropna(subset=["AccountID"])

#Merging the business account data with the loan data according to the mappings
MergedDataset = pd.merge(MergedBusinessesAndForecasts, LoanData, on="AccountID", how="inner")

#Selecting the required columns from the merged dataset and writing it into a CSV
MergedDataset = MergedDataset[["AccountID", "Starting Amount", "Ending Amount", "Net Change over Forecast Period", "Highest Amount", "Lowest Amount",
                               "Difference", "primary_sector", "company_type", "2019_revenue", "costs", "accounts_receivable",
                               "capital_and_reserves", "current_assets", "current_liabilities", "fixed_assets", "long_term_liabilities",
                               "provisions_for_liabilities", "loanAmount", "yearsOfCreditHistory", "totalCreditLines", "openCreditLines", "bankruptcies",
                               "delinquencies", "netValue", "status"]]
MergedDataset.to_csv("Merged Dataset (Whole Dataset, Assumption Set 3).csv")
