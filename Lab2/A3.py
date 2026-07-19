import openpyxl
import numpy as np
import time

def loadData():
    df = openpyxl.load_workbook("data.xlsx")
    df1 = df["IRCTC Stock Price"]
    mat=[]
    for row in range(0, 250):
        rows=[]
        for col in df1.iter_cols(1, 9):
            x = col[row].value
            rows.append(x)
            
        mat.append(rows)
    return mat

def extractPrices(mat):
    return np.array([mat[i][3] for i in range(1,len(mat))])

def extractWedPrices(mat):
    matNew = []
    [matNew.append(mat[i][3]) if mat[i][2]=="Wed" else 0 for i in range(1,len(mat))]
    return np.array(matNew)

def extractTups(mat):
    matNew = []
    matNew2 = []
    [(matNew.append(mat[i][8]),matNew2.append(mat[i][2])) for i in range(1,len(mat))]
    return np.array(matNew),np.array(matNew2)

def extractAprPrices(mat):
    matNew = []
    [matNew.append(mat[i][3]) if mat[i][1]=="Apr" else 0 for i in range(1,len(mat))]
    return np.array(matNew)

def probLoss(mat):
    matNew = [1 if mat[i][8]<0 else 0 for i in range(1,len(mat))]
    return sum(matNew)/len(matNew)

def probProfit(mat):
    matNew = [1 if mat[i][8]>0 else 0 for i in range(1,len(mat))]
    return sum(matNew)/len(matNew)

def probProfitWed(mat):
    matNew = []
    [matNew.append(1) if (mat[i][8]>0 and mat[i][2]=="Wed") else matNew.append(0) if mat[i][2]=="Wed" else 0 for i in range(1,len(mat))]
    return sum(matNew)/len(matNew)

def probProfitOnWed(mat):
    matNew = []
    [matNew.append(1) if (mat[i][8]>0 and mat[i][2]=="Wed") else matNew.append(0) if mat[i][2]=="Wed" else 0 for i in range(1,len(mat))]
    return sum(matNew)/(len(mat)-1)


def uMean(X):
    return sum(X)/len(X)

def uVar(X,m):
    return (sum([(i - m)**2 for i in X]))/len(X)

def plotScat(x,y):
    plt.scatter(x, y, color='blue', marker='o', s=100, alpha=0.8)
    plt.title("change vs day of the week")
    plt.xlabel("chg %")
    plt.ylabel("day of the week")
    plt.show()

# A3

mat = loadData()
Prices = extractPrices(mat)
mm = np.mean(Prices)
va = np.var(Prices)
print(f"Mean of the prices: {mm}")
print(f"Variance of the prices: {va}")
print()


complM = []
for i in range(0,10):
    t1 = time.time()
    uMean(Prices)
    t2 = time.time()
    complM.append(t2-t1)

m = uMean(Prices)
complV = []
for i in range(0,10):
    t1 = time.time()
    uVar(Prices,m)
    t2 = time.time()
    complV.append(t2-t1)

complexityM = uMean(complM)
complexityV = uMean(complV)
varr = uVar(Prices,m)

print(f"User Mean of the prices: {m}")
print(f"User Variance of the prices: {varr}")
print(f"Difference in: Means={mm-m}, Variances={va-varr}")
print()
print(f"Computational complexity of user defined mean function: {complexityM}")
print(f"Computational complexity of user defined variance function: {complexityV}")

print()

sampleMat = extractWedPrices(mat)
print(f"Mean of price data for all Wednesday: {uMean(sampleMat)}")
print(f"Difference b/w population mean and sample mean (population-sample): {uMean(sampleMat)-m}")

print()
sampleMat = extractAprPrices(mat)
print(f"Mean of price data for the April month: {uMean(sampleMat)}")
print(f"Difference b/w population mean and sample mean (population-sample): {uMean(sampleMat)-m}")

print()
print(f"Probability of making a loss over the IRCTC stock is {probLoss(mat)}")

numWed = len(extractWedPrices(mat))
print()
print(f"Probability of making a profit over the IRCTC stock on wednesday is {(probProfitOnWed(mat))}")


print()
print(f"Conditional probability of making profit, \ngiven that today is Wednesday is {probProfitWed(mat)}")

import matplotlib.pyplot as plt
X,Y = extractTups(mat)
plotScat(X,Y)
