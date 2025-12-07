#Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Reading in the results
Data = pd.read_csv("Cumulative Forecast Whole Dataset Results.csv")

#Denoting x and y values
x = np.arange(len(Data["Model"]))
y1 = Data["Mean SI"]
y2 = Data["Median SI"]

#Denoting bar width
Width = 0.35

#Visualising the results
plt.bar(x - Width/2, y1, Width, label="Mean SI")
plt.bar(x + Width/2, y2, Width, label="Median SI")
plt.xticks(x, Data["Model"])
plt.xlabel("Model")
plt.ylabel("Scatter Index (%)")
plt.title("Results of Cumulative Forecasts for Overall Dataset")
plt.legend()
plt.show()
