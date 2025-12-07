#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Reading in the data
Data = pd.read_csv("Assumption Set 2 Classification Results (Whole Dataset).csv")

#Declaring X and Y columns
x = np.arange(len(Data["Variables"]))
y1 = Data["Precision"]
y2 = Data["Recall"]

#Declaring bar width
Width = 0.35

#Visualising results
plt.bar(x - Width/2, y1, Width, label="Average Precision")
plt.bar(x + Width/2, y2, Width, label="Average Recall")
plt.xticks(x, Data["Variables"])
plt.xlabel("Variable Subset")
plt.ylabel("Precision or Recall Score (%)")
plt.title("Results for Assumption Set 2")
plt.legend()
plt.show()
