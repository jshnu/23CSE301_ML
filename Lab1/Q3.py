import copy

def MatPOW(mat,a):
    nMat = copy.deepcopy(mat)
    for _ in range(a-1):
        new = []
        for i in range(len(mat)):
            mm = []
            for j in range(len(mat)):
                ele = []
                for k in range(len(mat)):
                    ele.append(nMat[i][k]*mat[k][j])
                mm.append(sum(ele))
            new.append(mm)
        nMat = new
    return nMat
                
N = int(input("Enter the order of the square matrix: "))
m = int(input("Enter the value of m: "))
matA=[]
for i in range(N):
    row = []
    for j in range(N):
        x = int(input(f"Enter the value of element ({i},{j}) in the matrix: "))
        row.append(x)
    matA.append(row)

newMatA = MatPOW(matA,m)

print("Matrix A^m:")
for i in range(N):
    for j in range(N):
        print(newMatA[i][j],end=" ")
    print()
