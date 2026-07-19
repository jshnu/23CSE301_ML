import openpyxl
import numpy as np

def loadData():
    df = openpyxl.load_workbook("data.xlsx")
    df1 = df.active
    mat=[]
    for row in range(0, 11):
        rows=[]
        for col in df1.iter_cols(1, 5):
            x = col[row].value
            rows.append(x)
            
        mat.append(rows[1:])

    mat = mat[1:]
    return mat

def createMats(mat):
    X = []
    Y = []
    for i in range(len(mat)):
        X.append(mat[i][0:len(mat[0])-1])
        Y.append(mat[i][len(mat[0])-1:len(mat[0])])
    return np.array(X),np.array(Y)

def calc(X,Y):
    pinvX = np.linalg.pinv(X[0:3])
    return np.dot(pinvX[0:3],Y[0:3])

def mark(Y):
    for i in range(0,10):
        print(f"C{i+1} is",end=" ")
        if(Y[i][0]>200):
            print(f"Rich (Spent {Y[i][0]}rs)")
        else:
            print(f"Poor (Spent {Y[i][0]}rs)")


# A1
mat = loadData()


X,Y = createMats(mat)
print(X)
print("Rank of matrix X:",np.linalg.matrix_rank(X))

print(Y)
print("Rank of matrix Y:",np.linalg.matrix_rank(Y))


costOfEach = calc(X,Y)

print(f"Cost per: Candy={costOfEach[0][0]:.2f}rs, Mango={costOfEach[1][0]:.2f}rs, Milk Packet={costOfEach[2][0]:.2f}rs")
print()
# A2

mark(Y)
