#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt

#Reading in the data and splitting it into repayers and defaulters
Data = pd.read_csv("../Menna Dissertation Dataset/MSc project/open_src_data/sme_loans/sme_loans_data.csv")
RepaidData = Data[Data["status"] == "Paid"]
DefaultedData = Data[Data["status"] == "Defaulted"]
Data = pd.concat([RepaidData, DefaultedData], axis=0)

#Visualising the class distribution in a pie chart
plt.pie(Data["status"].value_counts(), labels=["Paid", "Defaulted"], autopct="%1.1f%%")
plt.title("Distribution of Repayment and Default in SME Loans Dataset")
plt.show()
