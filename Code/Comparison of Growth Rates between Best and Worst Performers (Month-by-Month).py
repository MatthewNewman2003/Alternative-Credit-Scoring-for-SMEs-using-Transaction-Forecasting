#Importing libraries
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

#Reading in the data and extracting account IDs
BoostedTransactions = pd.read_csv("Boosted Transaction Dataset.csv")
AccountIDs = BoostedTransactions["AccountId"].values

#Converting transaction dates into months
BoostedTransactions["TransactionDate"] = pd.to_datetime(BoostedTransactions["TransactionDate"])
BoostedTransactions["TransactionDate"] = BoostedTransactions["TransactionDate"].dt.to_period("M").dt.to_timestamp()

#Grouping transactions by account and month and summating the net transaction amount per grouping
GroupedSumOfTransactionsByAccountAndMonth = BoostedTransactions.groupby(by=["AccountId", "TransactionDate"]).agg(["sum"])["Amount"]

#Declaring the 10 poorest account IDs and creating a list to store their recent growth percentage figures
PoorAccountIDs = ["16443", "16429", "16424", "78db5423-a74c-44ba-b1db-26f8b3e24014", "d3886a65-04e5-4575-b798-7e217cd7d3c6",
                  "2aaef9e1-6293-4f85-97cf-dc9ef8d6ae19", "1186c2fd-7a00-4f21-956f-8c5a896d9111", "16461", "43aaf6de-65e5-4d44-876d-c9bb99722e59",
                  "ef9951b1-7a6c-450b-bbcd-eef377be520d"]
PoorGrowthPercentages = []

GroupedSumOfTransactionsByAccountAndMonth = GroupedSumOfTransactionsByAccountAndMonth.reset_index()

#For each poor account, the data is split into training and testing sets
#The final 3 percentage growth rates are then taken, averaged and put into the percentages list
for i in PoorAccountIDs:
    AccountID = i
    GrowthPercentages = []

    ForecastingTestTimeSeries = GroupedSumOfTransactionsByAccountAndMonth[GroupedSumOfTransactionsByAccountAndMonth["AccountId"] == AccountID][["TransactionDate", "sum"]]

    ForecastingTestTimeSeries = ForecastingTestTimeSeries.set_index("TransactionDate")

    ForecastingTestTimeSeries = ForecastingTestTimeSeries.resample("MS").sum()

    TrainEnd = int(0.7 * len(ForecastingTestTimeSeries))

    TrainData = ForecastingTestTimeSeries[:TrainEnd]
    TestData = ForecastingTestTimeSeries[TrainEnd:]

    KeyIndices = TrainData[-4:]

    Mean = abs(np.mean(KeyIndices))
    for j in range(0, len(KeyIndices)):
        if j == 0:
            pass
        else:
            PreviousRecord = KeyIndices.iloc[j-1]
            CurrentRecord = KeyIndices.iloc[j]
            Change = CurrentRecord - PreviousRecord
            Percentage = abs((Change/PreviousRecord)*100)
            GrowthPercentages.append(Percentage)

    MeanPercentage = np.mean(GrowthPercentages)
    PoorGrowthPercentages.append(MeanPercentage)

print(PoorGrowthPercentages)

#Declaring the good performers and creating a list to store their recent growth percentages
GoodAccountIDs = ["4b171653-23c9-4f55-924a-46e51bf11ee4", "196d06cf-e811-4aab-9649-ab6642551b9b", "eaccaa73-b149-48ae-a224-2f231f46432c",
                  "ee64480e-4c31-4c7c-9858-91cd2ae43e06", "0734342c-6b9a-42f8-834c-08a98d60e98c", "77303cad-e32f-4568-9482-552625c1bf54",
                  "e8ebaf0f-ea95-46a1-ae7d-9d00d68960c9", "265754d8-dff8-4d8c-a663-a913ecb071a9", "8f26459d-4da4-4c12-b2a1-56cb3f2559b4",
                  "6d990843-e957-4d5d-8c33-85a8026a07d0"]                  
GoodGrowthPercentages = []

#For each good account, the data is split into training and testing sets
#The final 3 percentage growth rates in the training set are then taken, averaged and put into the percentages list
for i in GoodAccountIDs:
    AccountID = i
    GrowthPercentages = []

    ForecastingTestTimeSeries = GroupedSumOfTransactionsByAccountAndMonth[GroupedSumOfTransactionsByAccountAndMonth["AccountId"] == AccountID][["TransactionDate", "sum"]]

    ForecastingTestTimeSeries = ForecastingTestTimeSeries.set_index("TransactionDate")

    ForecastingTestTimeSeries = ForecastingTestTimeSeries.resample("MS").sum()

    TrainEnd = int(0.7 * len(ForecastingTestTimeSeries))

    TrainData = ForecastingTestTimeSeries[:TrainEnd]
    TestData = ForecastingTestTimeSeries[TrainEnd:]

    KeyIndices = TrainData[-4:]

    Mean = abs(np.mean(KeyIndices))
    for j in range(0, len(KeyIndices)):
        if j == 0:
            pass
        else:
            PreviousRecord = KeyIndices.iloc[j-1]
            CurrentRecord = KeyIndices.iloc[j]
            Change = CurrentRecord - PreviousRecord
            Percentage = abs((Change/PreviousRecord)*100)
            GrowthPercentages.append(Percentage)

    MeanPercentage = np.mean(GrowthPercentages)
    GoodGrowthPercentages.append(MeanPercentage)

print(GoodGrowthPercentages)

#Performing the one-tailed t-test to determine if the percentage growth rates are higher on poorer performing accounts
TestStatistic, PValue = ttest_ind(PoorGrowthPercentages, GoodGrowthPercentages)
print("Test statistic:",TestStatistic)
print("P-Value:",(PValue/2))
