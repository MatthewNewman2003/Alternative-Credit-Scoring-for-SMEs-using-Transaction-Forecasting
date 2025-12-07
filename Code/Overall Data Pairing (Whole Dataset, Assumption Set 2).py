#Importing libraries
import pandas as pd
import random
from datetime import datetime
import datetime as dt
from dateutil import parser
from scipy.spatial import KDTree
import numpy as np

#Reading in the data and filtering the loan data to only include repayers and defaulters
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

#Calculating the number of unique business accounts and unique transaction accounts and resultantly determining the maximum number of feasible pairs
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

#Grouping transactions by account/month and account and calculating the net transaction amount per grouping
GroupedSumOfTransactionsByAccountAndMonth = TransactionData.groupby(by=["AccountId", "TransactionDate"]).agg(["sum"])["Amount"]
print(GroupedSumOfTransactionsByAccountAndMonth)

GroupedSumOfTransactionsByAccount = TransactionData.groupby(by="AccountId").agg(["sum"])["Amount"]
print(GroupedSumOfTransactionsByAccount)

GroupedSumOfTransactionsByAccount = GroupedSumOfTransactionsByAccount.reset_index(drop=False)
print(GroupedSumOfTransactionsByAccount)

#Calculating each business account's cash flow (revenue-costs)
BusinessAccountData["Cash Flow"] = BusinessAccountData["2019_revenue"] - BusinessAccountData["costs"]

#Working out the cash flow and transaction amount percentiles for the business accounts and transaction accounts
BusinessAccountData["Cash Flow Percentile"] = BusinessAccountData["Cash Flow"].rank(pct=True)
GroupedSumOfTransactionsByAccount["Transaction Amount Percentile"] = GroupedSumOfTransactionsByAccount["sum"].rank(pct=True)

print(BusinessAccountData['Cash Flow Percentile'].isnull().sum())
print(GroupedSumOfTransactionsByAccount['Transaction Amount Percentile'].isnull().sum())

BusinessAccountData['Cash Flow'] = BusinessAccountData['Cash Flow'].fillna(0)
BusinessAccountData["Cash Flow Percentile"] = BusinessAccountData["Cash Flow"].rank(pct=True)

#Performing percentile-based mapping of business accounts and transaction accounts based on transaction amount and cash flow percentiles
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

#Pairing the business account and transaction datasets based on the mappings
NumberToIDMappings = {}
for i in range(0, len(MatchedDataFrame)):
    AccountNumber = MatchedDataFrame.iloc[i]["Current Account Number"]
    AccountID = MatchedDataFrame.iloc[i]["Matched AccountID"]
    NumberToIDMappings[AccountNumber] = AccountID

BusinessAccountData["AccountID"] = BusinessAccountData["current_account_number"].map(NumberToIDMappings)
BusinessAccountData = BusinessAccountData.dropna(subset=["AccountID"])

#Reading in the cumulative and month-by-month forecast datasets
CumulativeForecastData = pd.read_csv("Naive Forecasts (Cumulative).csv")
MonthByMonthForecastData = pd.read_csv("SSA Forecasts (Month-by-Month).csv")

#Making sure the account IDs are interpretable
BusinessAccountData["AccountID"] = BusinessAccountData["AccountID"].astype("str")
BusinessAccountData["AccountID"] = BusinessAccountData["AccountID"].astype("object")
BusinessAccountData["AccountID"] = BusinessAccountData["AccountID"].astype(str).str.strip()
CumulativeForecastData["AccountID"] = CumulativeForecastData["AccountID"].astype(str).str.strip()
MonthByMonthForecastData["AccountID"] = MonthByMonthForecastData["AccountID"].astype(str).str.strip()

#Merging the forecasts with the business accounts
MergedBusinessesAndForecasts = pd.merge(BusinessAccountData, CumulativeForecastData, on="AccountID", how="inner")
MergedBusinessesAndForecasts = pd.merge(MergedBusinessesAndForecasts, MonthByMonthForecastData, on="AccountID", how="inner")
print(MergedBusinessesAndForecasts)

print(LoanData)

#Declaring a random seed for reproducibility and declaring today's date
np.random.seed(43)
Today = datetime.today()

#Working out how many years each business has been in business for
IncorporationDates = MergedBusinessesAndForecasts["incorporation_date"]

ParsedIncorporationDates = [parser.parse(date) for date in IncorporationDates]

MergedBusinessesAndForecasts["incorporation_date"] = ParsedIncorporationDates

MergedBusinessesAndForecasts["Years in Business"] = (Today - MergedBusinessesAndForecasts["incorporation_date"]).dt.days / 365.25

#Determining time percentile ranks for the business account and loan datasets based on years in business and years of credit history respectively
MergedBusinessesAndForecasts["Time Percentile Rank"] = MergedBusinessesAndForecasts["Years in Business"].rank(pct=True)
LoanData["Time Percentile Rank"] = LoanData["yearsOfCreditHistory"].rank(pct=True)

MergedBusinessesAndForecasts = MergedBusinessesAndForecasts.sort_values("Time Percentile Rank").reset_index(drop=True)
LoanData = LoanData.sort_values("Time Percentile Rank").reset_index(drop=True)

#Pairing business accounts with loan accounts under the assumption that older businesses will have a longer credit history
UsedBusinesses = set()
BusinessToLoanMatches = []

for IndexLoan, RowLoan in LoanData.iterrows():
    UnusedBusinesses = MergedBusinessesAndForecasts[~MergedBusinessesAndForecasts["AccountID"].isin(UsedBusinesses)]
    
    Differences = (UnusedBusinesses["Time Percentile Rank"] - RowLoan["Time Percentile Rank"]).abs()
    MinimumDifference = Differences.min()
    
    CandidateIndices = Differences[Differences == MinimumDifference].index.tolist()
    ChosenIndex = np.random.choice(CandidateIndices)
    
    BusinessToLoanMatches.append({
        "Business Account ID" : MergedBusinessesAndForecasts.loc[ChosenIndex, "AccountID"],
        "Incorporation Date" : MergedBusinessesAndForecasts.loc[ChosenIndex, "incorporation_date"],
        "Years in Business" : MergedBusinessesAndForecasts.loc[ChosenIndex, "Years in Business"],
        "Business Time Percentile Rank" : MergedBusinessesAndForecasts.loc[ChosenIndex, "Time Percentile Rank"],
        "Loan Account ID" : RowLoan["accountId"],
        "Years of Credit History" : RowLoan["yearsOfCreditHistory"],
        "Loan Time Percentile Rank" : RowLoan["Time Percentile Rank"]
    })
    
    UsedBusinesses.add(MergedBusinessesAndForecasts.loc[ChosenIndex, "AccountID"])

#Pairing business accounts to loan accounts according to the pairings generated above and merging the datasets
BusinessAndLoanPairings = pd.DataFrame(BusinessToLoanMatches)


LoanToBusinessMappings = {}
for i in range(0, len(BusinessAndLoanPairings)):
    BusinessAccountID = BusinessAndLoanPairings.iloc[i]["Business Account ID"]
    LoanAccountID = BusinessAndLoanPairings.iloc[i]["Loan Account ID"]
    LoanToBusinessMappings[LoanAccountID] = BusinessAccountID

LoanData["AccountID"] = LoanData["accountId"].map(LoanToBusinessMappings)
LoanData = LoanData.dropna(subset=["AccountID"])
MergedDataset = pd.merge(MergedBusinessesAndForecasts, LoanData, on="AccountID", how="inner")

#Selecting the required columns from the merged dataset and writing it to a CSV
MergedDataset = MergedDataset[["AccountID", "Starting Amount", "Ending Amount", "Net Change over Forecast Period", "Highest Amount", "Lowest Amount",
                               "Difference", "primary_sector", "company_type", "2019_revenue", "costs", "accounts_receivable",
                               "capital_and_reserves", "current_assets", "current_liabilities", "fixed_assets", "long_term_liabilities",
                               "provisions_for_liabilities", "loanAmount", "yearsOfCreditHistory", "totalCreditLines", "openCreditLines", "bankruptcies",
                               "delinquencies", "netValue", "status"]]
MergedDataset.to_csv("Merged Dataset (Whole Dataset, Assumption Set 2).csv")
