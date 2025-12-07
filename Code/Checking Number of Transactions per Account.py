#Importing libraries
import pandas as pd
from datetime import datetime
from dateutil import parser
import matplotlib.pyplot as plt

#Reading in the data
Data = pd.read_csv("synthetic_ob_transactions.csv")

#Visualising number of transactions per account in a boxplot
BusinessAccountAmounts = Data["AccountId"].value_counts()
plt.boxplot(BusinessAccountAmounts)
plt.title("Distribution of Number of Transactions per Account")
plt.ylabel("Number of Transactions")
plt.show()
