#Importing libraries
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

#Reading in the data
BoostedTransactions = pd.read_csv("Boosted Transaction Dataset.csv")
AccountIDs = BoostedTransactions["AccountId"].values

#Converting transaction dates to months
BoostedTransactions["TransactionDate"] = pd.to_datetime(BoostedTransactions["TransactionDate"])
BoostedTransactions["TransactionDate"] = BoostedTransactions["TransactionDate"].dt.to_period("M").dt.to_timestamp()

#Grouping transactions by account and month and summating the net transaction amount per grouping
GroupedSumOfTransactionsByAccountAndMonth = BoostedTransactions.groupby(by=["AccountId", "TransactionDate"]).agg(["sum"])["Amount"]

#Declaring a list of poorly performing accounts and a list to store their averaged growth percentages in
PoorAccountIDs = ["997ef1f2-371c-4607-8758-54ff96ac6e02", "8ce8dbb8-1682-491c-bd22-455fffefabed", "16424", "0a6729fb-4fe2-4987-921e-8ab6a6f289cb",
                  "16429", "c3beeaa8-87ce-455d-9482-e01679b9d1eb", "b1a8e3ff-b811-4629-b37f-c303b4094e2c", "3a98e91c-b44d-4427-ba16-8b762379db5c",
                  "efb8bcab-7a10-45d9-b6b3-5a2f855e2c96", "703f718c-79ac-42e9-a793-37e1795e7ef8"]
PoorGrowthPercentages = []

ResetDataFrame = GroupedSumOfTransactionsByAccountAndMonth.reset_index()

#For each poor account, the cumulative sum of transactions over time is calculated
#This data is then split into training and testing sets
#The final three percentage growth rates in the training set are averaged and appended into the growth percentages list
for i in PoorAccountIDs:
    AccountID = i
    GrowthPercentages = []

    TimeSeries = ResetDataFrame[ResetDataFrame["AccountId"] == AccountID]
    TimeSeries = TimeSeries.set_index("TransactionDate")

    TimeSeries = TimeSeries.resample("MS").sum()
    for j in range(0, len(TimeSeries)):
        if TimeSeries.iloc[j]["AccountId"] == 0:
          TimeSeries.at[TimeSeries.index[j], "AccountId"] = AccountID

    NewGroupedDataset = TimeSeries
    NewGroupedDataset = NewGroupedDataset.reset_index()
    NewGroupedDataset = NewGroupedDataset.set_index(["AccountId", "TransactionDate"])

    GroupedCumulativeSumOfTransactionsByAccountAndMonth = NewGroupedDataset.groupby(level=0).cumsum().reset_index()

    ForecastingTestTimeSeries = GroupedCumulativeSumOfTransactionsByAccountAndMonth[GroupedCumulativeSumOfTransactionsByAccountAndMonth["AccountId"] == AccountID][["TransactionDate", "sum"]]

    ForecastingTestTimeSeries = ForecastingTestTimeSeries.set_index("TransactionDate")

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

#Declaring a list of well performing accounts and a list to store their averaged growth percentages in
GoodAccountIDs = ["16434", "16453", "3fbc9c73-8f31-4391-83fd-bb0161aaff53", "4504a897-bbdb-46c5-932b-e6109adcd825", "e3224f97-0eb1-4f42-963e-87bb423bdab7",
                  "47255e34-7fcb-457c-aee6-fe0b48bae3c7", "2971bf3e-670d-4e90-b127-921b7dae0f24", "c9f19b41-8752-401e-b054-12f9d14cdeb7",
                  "16450", "16c0e231-2dd6-42f6-908a-a668c98c9c21"]                  
GoodGrowthPercentages = []

#For each poor account, the cumulative sum of transactions over time is calculated
#This data is then split into training and testing sets
#The final three percentage growth rates in the training set are averaged and appended into the growth percentages list
for i in GoodAccountIDs:
    AccountID = i
    GrowthPercentages = []

    TimeSeries = ResetDataFrame[ResetDataFrame["AccountId"] == AccountID]
    TimeSeries = TimeSeries.set_index("TransactionDate")

    TimeSeries = TimeSeries.resample("MS").sum()
    for j in range(0, len(TimeSeries)):
        if TimeSeries.iloc[j]["AccountId"] == 0:
          TimeSeries.at[TimeSeries.index[j], "AccountId"] = AccountID

    NewGroupedDataset = TimeSeries
    NewGroupedDataset = NewGroupedDataset.reset_index()
    NewGroupedDataset = NewGroupedDataset.set_index(["AccountId", "TransactionDate"])

    GroupedCumulativeSumOfTransactionsByAccountAndMonth = NewGroupedDataset.groupby(level=0).cumsum().reset_index()

    ForecastingTestTimeSeries = GroupedCumulativeSumOfTransactionsByAccountAndMonth[GroupedCumulativeSumOfTransactionsByAccountAndMonth["AccountId"] == AccountID][["TransactionDate", "sum"]]

    ForecastingTestTimeSeries = ForecastingTestTimeSeries.set_index("TransactionDate")

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

#Performing the one-tailed t-test to determine whether the growth rate percentages are higher among the poor performers
TestStatistic, PValue = ttest_ind(PoorGrowthPercentages, GoodGrowthPercentages)
print("Test statistic:",TestStatistic)
print("P-Value:",(PValue/2))
