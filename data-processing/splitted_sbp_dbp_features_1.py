#G. Thambiraj, U. Gandhi, U. Mangalanathan, V. J. M. Jose, and M. Anand, “Investigation on the effect of womersley number, ecg and ppg features for cuff less blood pressure estimation using machine learning,” Biomedical Signal Processing and Control, vol. 60, p. 101942, 2020.
import numpy as np
from matplotlib import pyplot as plt

def getFeatures_SBP_DBP(ppg_only = False, phy_only = False):
    data = np.loadtxt("all_ppg_features.csv", delimiter=",", dtype=np.double)
    if(ppg_only):
        data = np.loadtxt("ppg_only_features.csv", delimiter=",", dtype=np.double)
    elif(phy_only):
        data = np.loadtxt("phy_only_features.csv", delimiter=",", dtype=np.double)
    print(data.shape)

    sbp_column_index = -2
    dbp_column_index = -1
    count = 0
    valid_rows =[]

    for i in range(data.shape[0]):
        row = data[i]
        is_valid = True
        if row[sbp_column_index]<60 or row[dbp_column_index]<40 or row[sbp_column_index]<row[dbp_column_index] or row[sbp_column_index]>200 or row[dbp_column_index]>160:
            is_valid = False

        for col in row:
            if np.isnan(col) or np.isinf(col):
                is_valid = False
                break
        if is_valid:
            valid_rows.append(count)
        count += 1

    cleaned_data = np.zeros((len(valid_rows), data.shape[1]))

    count = 0
    for row in valid_rows:
        cleaned_data[count] = data[row]
        count += 1

    data = np.array(cleaned_data)
    sbp = np.array(data[:,sbp_column_index], dtype=np.float64)
    dbp = np.array(data[:,dbp_column_index], dtype=np.float64)
    features = data[:,0:data.shape[1]-2]
    labels = np.zeros(sbp.shape[0], dtype=np.int64) #changed
    hypertensive = np.zeros(sbp.shape[0], dtype=np.int64)
    targets = np.zeros(sbp.shape[0], dtype=np.int64)

    for i in range(sbp.shape[0]):
        sbp_val = sbp[i]
        dbp_val = dbp[i]
   
        if(sbp_val<120 and dbp_val<80):
            labels[i] = 1
        elif(sbp_val>=120 and sbp_val<= 129 and dbp_val<80):
            labels[i] = 2
        elif((sbp_val>129 and sbp_val<= 139) or (dbp_val>=80 and dbp_val<=89)):
            labels[i] = 3
        elif((sbp_val>139 and sbp_val<= 179) or (dbp_val>89 and dbp_val<=119)):
            labels[i] = 4
        elif(sbp_val>179 or dbp_val>=119):
            labels[i] = 5

    print(features.shape)
    print(sbp)
    print(dbp)
    return {
        "features": features,
        "sbp": sbp,
        "dbp": dbp,
        "one-hot-labels": labels,
        "targets": targets,
        "hypertensive": hypertensive
    }

def normalize(X):
    mu = np.mean(X, 0)
    sigma = np.std(X, 0)
    X = (X-mu) / sigma
    return X

def add_intercept(X):
    X = np.hstack((np.ones((X.shape[0],1)),X))
    return X

d = getFeatures_SBP_DBP()
s = d["sbp"]
d = d["dbp"]
ds = d - s
count = 0

for v in ds:
    if v >= 0:
        count += 1
print(count)
print(np.min(s), np.min(d))

count = 0

for v in s:
    if v < 90:
        count += 1
print(count)

count = 0
for v in d:
    if v < 60:
        count += 1
print(count)
print(np.sum(getFeatures_SBP_DBP()["hypertensive"])/getFeatures_SBP_DBP()["hypertensive"].shape[0])