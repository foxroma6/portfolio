## competition description

## Import required libraries
import pandas as pd
import numpy as np
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder

## load train data and test dataset / combine two dataset
train_dt = pd.read_csv("C:/Users/HOSUB/Desktop/auction_master/Auction_master_train_en.csv")
test_dt = pd.read_csv("C:/Users/HOSUB/Desktop/auction_master/Auction_master_test_en.csv")
combine = [train_dt,test_dt]

## check dataset content
#combine.head()
#combine.tail()
#combine.describe()

## check dataset type
##combine.info()

## drop unnecessary columns
train_dt1 = train_dt.drop(['Auction_key', 'Appraisal_company',
                           'Final_result','addr_etc','Specific','Close_result',
                           'point.x','point.y', 'Close_date', 'Preserve_regist_date'], axis=1)
test_dt1 = test_dt.drop(['Auction_key', 'Appraisal_company',
                           'Final_result','addr_etc','Specific','Close_result',
                           'point.x','point.y', 'Close_date', 'Preserve_regist_date'], axis=1)

#### Check if there are any NULL values in Train Data
print("Total Train Features with NaN Values = " 
      + str(train_dt1.columns[train_dt1.isnull().sum() != 0].size))
# Nan value -> 0

print("Total Test Features with NaN Values = " 
      + str(train_dt1.columns[test_dt1.isnull().sum() != 0].size))
# Nan value -> 0

## convert object column values to binary values
## 1. Bid_class
train_dt1['Bid_class'].unique()
Bid_class_mapping = {'individual': 1 , 'general': 2}
train_dt1['Bid_class'] = train_dt1['Bid_class'].map(Bid_class_mapping)
train_dt1['Bid_class'] = train_dt1['Bid_class'].fillna(0)

## 2. Creditor
train_dt1['Creditor'].unique()
Creditor_mapping = {'Company or Group': 1, 'private': 2}
train_dt1['Creditor'] = train_dt1['Creditor'].map(Creditor_mapping)
train_dt1['Creditor'] = train_dt1['Creditor'].fillna(0)

## 3. Apartment_usage
train_dt1['Apartment_usage'].unique()
Apartment_usage_mapping = {'Apartment & stores': 1, 'Apartment': 2}
train_dt1['Apartment_usage'] = train_dt1['Apartment_usage'].map(Apartment_usage_mapping)
train_dt1['Apartment_usage'] = train_dt1['Apartment_usage'].fillna(0)

## 4. Share_auction_YorN
train_dt1['Share_auction_YorN'].unique()
Share_auction_YorN_mapping = {'N': 1, 'Y': 2}
train_dt1['Share_auction_YorN'] = train_dt1['Share_auction_YorN'].map(Share_auction_YorN_mapping)
train_dt1['Share_auction_YorN'] = train_dt1['Share_auction_YorN'].fillna(0)

## 5. Auction_class
train_dt1['Auction_class'].unique()
Bid_class_mapping = {'random': 1 , 'forced': 2}
train_dt1['Auction_class'] = train_dt1['Auction_class'].map(Bid_class_mapping)
train_dt1['Auction_class'] = train_dt1['Auction_class'].fillna(0)

## convert date value into month value
train_dt1 = train_dt1.assign(Appraisal_mth = train_dt1['Appraisal_date'].str.split("-", n=2, expand=True)[0] + train_dt1['Appraisal_date'].str.split("-", n=2, expand=True)[1])
train_dt1 = train_dt1.drop(['Appraisal_date'], axis=1)
train_dt1["Appraisal_mth"] = train_dt1["Appraisal_mth"].astype(str).astype(int)

## create new column by calculating end - start date value
train_dt1.Final_auction_date = pd.to_datetime(train_dt1.Final_auction_date)
train_dt1.First_auction_date = pd.to_datetime(train_dt1.First_auction_date)

train_dt1 = train_dt1.assign(Auction_day_diff=train_dt1.Final_auction_date-train_dt1.First_auction_date)
train_dt1 = train_dt1.drop(['Final_auction_date', 'First_auction_date'], axis=1)
train_dt1["Auction_day_diff"] = train_dt1["Auction_day_diff"].astype(str).str.split(" ",expand=True,n=2)[0].astype(int)

## split addr and convert this column into dummy values
train_dt1 = train_dt1.assign(Addr_City = train_dt1['addr_en'].str.split().str[-1])
#train_dt1 = train_dt1.assign(Addr_Street = train_dt1['addr_en'].str.split(" ",expand=True,n=4)[0])
#train_dt1 = train_dt1.assign(Addr_District = train_dt1['addr_en'].str.split(" ",expand=True,n=2)[2].str.split("-",expand=True,n=2)[0])
train_dt1 = train_dt1.drop(['addr_en'], axis=1)

train_dt1 = pd.get_dummies(train_dt1, columns=["Addr_City"])
#train_dt1 = pd.get_dummies(train_dt1, columns=["Addr_Street"])
#train_dt1 = pd.get_dummies(train_dt1, columns=["Addr_District"])

## convert object column values to binary values
## 1. Bid_class
test_dt1['Bid_class'].unique()
Bid_class_mapping = {'individual': 1 , 'general': 2}
test_dt1['Bid_class'] = test_dt1['Bid_class'].map(Bid_class_mapping)
test_dt1['Bid_class'] = test_dt1['Bid_class'].fillna(0)

## 2. Creditor
test_dt1['Creditor'].unique()
Creditor_mapping = {'Company or Group': 1, 'private': 2}
test_dt1['Creditor'] = test_dt1['Creditor'].map(Creditor_mapping)
test_dt1['Creditor'] = test_dt1['Creditor'].fillna(0)

## 3. Apartment_usage
test_dt1['Apartment_usage'].unique()
Apartment_usage_mapping = {'Apartment & stores': 1, 'Apartment': 2}
test_dt1['Apartment_usage'] = test_dt1['Apartment_usage'].map(Apartment_usage_mapping)
test_dt1['Apartment_usage'] = test_dt1['Apartment_usage'].fillna(0)

## 4. Share_auction_YorN
test_dt1['Share_auction_YorN'].unique()
Share_auction_YorN_mapping = {'N': 1, 'Y': 2}
test_dt1['Share_auction_YorN'] = test_dt1['Share_auction_YorN'].map(Share_auction_YorN_mapping)
test_dt1['Share_auction_YorN'] = test_dt1['Share_auction_YorN'].fillna(0)

## 5. Auction_class
test_dt1['Auction_class'].unique()
Bid_class_mapping = {'random': 1 , 'forced': 2}
test_dt1['Auction_class'] = test_dt1['Auction_class'].map(Bid_class_mapping)
test_dt1['Auction_class'] = test_dt1['Auction_class'].fillna(0)

## convert date value into month value
test_dt1 = test_dt1.assign(Appraisal_mth = test_dt1['Appraisal_date'].str.split("-", n=2, expand=True)[0] + test_dt1['Appraisal_date'].str.split("-", n=2, expand=True)[1])
test_dt1 = test_dt1.drop(['Appraisal_date'], axis=1)
test_dt1["Appraisal_mth"] = test_dt1["Appraisal_mth"].astype(str).astype(int)

## create new column by calculating end - start date value
test_dt1.Final_auction_date = pd.to_datetime(test_dt1.Final_auction_date)
test_dt1.First_auction_date = pd.to_datetime(test_dt1.First_auction_date)

test_dt1 = test_dt1.assign(Auction_day_diff=test_dt1.Final_auction_date-test_dt1.First_auction_date)
test_dt1 = test_dt1.drop(['Final_auction_date', 'First_auction_date'], axis=1)
test_dt1["Auction_day_diff"] = test_dt1["Auction_day_diff"].astype(str).str.split(" ",expand=True,n=2)[0].astype(int)

## split addr and convert this column into dummy values
test_dt1 = test_dt1.assign(Addr_City = test_dt1['addr_en'].str.split().str[-1])
#test_dt1 = test_dt1.assign(Addr_Street = test_dt1['addr_en'].str.split(" ",expand=True,n=4)[0])
#test_dt1 = test_dt1.assign(Addr_District = test_dt1['addr_en'].str.split(" ",expand=True,n=2)[2].str.split("-",expand=True,n=2)[0])
test_dt1 = test_dt1.drop(['addr_en'], axis=1)

test_dt1 = pd.get_dummies(test_dt1, columns=["Addr_City"])
#test_dt1 = pd.get_dummies(test_dt1, columns=["Addr_Street"])
#test_dt1 = pd.get_dummies(test_dt1, columns=["Addr_District"])

print("Train set size: {}".format(train_dt1.shape))
print("Test set size: {}".format(test_dt1.shape))