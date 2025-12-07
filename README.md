# Alternative-Credit-Scoring-for-SMEs-using-Transaction-Forecasting
**Background**

This repository contains the code for my MSc dissertation at Cardiff University: Alternative 
Credit Scoring for SMEs using Transaction Forecasting.

Working in conjunction with Menna, I explored an alternative method of credit scoring for SMEs.
With many SMEs being locked out of credit by traditional models relying on credit history, my 
project aimed to explore an alternative credit scoring model for SMEs utilising transaction forecasting
with the aim of widening access to credit for often-excluded businesses.

The approach was underpinned by synthetic datasets, generated from a base dataset using Python’s
Synthetic Data Vault library. With Python libraries such as SciKit-Learn and Keras being employed,
the modelling approach encompassed two parts. Firstly, I tested an LSTM neural network, SSA, Prophet
and a naive forecast for transaction forecasting on both a cumulative and month-by-month basis. I then
took the strongest of these forecasting approaches for each and integrated variables from them into a
credit scoring classification model, powered by XGBoost.

If you wish to read the dissertation itself, it is contained within "Dissertation.pdf".

**Code Instructions**

The original datasets are contained within the folder entitled “Original Datasets”. The 
transaction dataset is entitled “synthetic_ob_transactions.csv”, the current accounts 
dataset is entitled “synthetic_uk_business_current_accounts.csv”, and the SME loans 
dataset entitled in “sme_loans_data.csv”. The post-processing datasets are contained 
in “Post-Processing Datasets”, the errors and forecasts CSVs from the forecasting 
models are in “Errors and Forecasts CSVs”, and the results CSVs are in “Results 
CSVs”.

Prior to running any forecasting models, the code for generating the boosted synthetic 
transaction dataset, entitled “Generating Synthetic Transaction Dataset.py”, needs to 
be executed. This will take the original transaction data, boost it to add 800 new 
accounts with corresponding transactions, and write the boosted dataset to a CSV, 
entitled “Boosted Transaction Dataset.csv”. This file will be needed to run all 
forecasting models on the whole dataset.

Prior to running the forecasting models for the longer subset, the code for generating 
the longer subset, entitled “Generating Subset of Longer Accounts from Synthetic 
Data (14 Months).py”, needs to be executed. This will take the boosted transaction 
data, filter it to only contain accounts with 14 or more months of transaction data, and 
write the longer subset to a CSV, entitled “Longer Subset of Transaction Data (14 
Months).csv”. This file will be needed to run all forecasting models on the longer 
subset.

Prior to running the SSA models, the Python class “mySSA.py” will be needed within 
the Google Colab environment. The file can be found in the code folder, or alternatively,
it can be downloaded from the original GitHub source: https://github.com/aj-cloete/pssa   

To run the LSTM models, accounts needed to be processed in batches of 150 so as not 
to violate Google Colab’s memory requirements. Each batch is shown as a separate 
code block in the notebook. Run all cells prior to any of the batches before running 
the batches themselves but only run the code generating the UniqueAccounts set and 
corresponding “UniqueAccounts.json” file before running the first batch (this is to 
maintain the same order of accounts between batches and ensure no duplicate account 
runs). After one batch has been run, press “Restart session” in Google Colab and re
execute all cells prior to the LSTM batches apart from the code generating the account 
list set and JSON file, as explained above. Only run the block for the batch you wish 
to run (e.g. run only accounts 0-150 first before restarting, then restart session and run 
only accounts 150-300 and so on). This is to prevent the running of all accounts from 
exceeding the Google Colab free tier’s memory limits.

When running each forecasting model’s notebook, two CSV files will be generated: 
“[Model Name] Errors.csv” and “[Model Name] Forecasts.csv”. The errors files are 
used to generate the error rates shown in the final code block of the Google Colab 
notebook, and the forecasts files should be downloaded, as they are needed later for 
the data pairing prior to use in classification. Due to them being the strongest 
performers in each case, the naïve forecasts were used for cumulative forecasting and 
the SSA forecasts were used for month-by-month forecasting, so these are the two 
necessary files. Only the whole dataset forecast files were needed for classification, 
however.

Before running the classification models, the file pairing the three disparate datasets, 
entitled “Overall Data Pairing [Assumption Set].py” needs to be executed for each 
assumption set. This performs the pairing according to the outlined assumptions for 
each assumption set, and generates a file entitled “Paired Dataset [Assumption 
Set].csv” for each assumption set. These files will be needed to run the classification 
models. 

