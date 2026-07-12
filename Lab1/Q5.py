import random

def mean(arr):
    return (sum(arr)/len(arr))

def modeSorted(arr):
    m = [arr[0],1]
    c = 0
    for i in range(1,len(arr)):
        if arr[i-1]==arr[i]:
            c+=1
            if m[1]<c:
                m = [arr[i],c]
        else:
            c=1
    return m[0]

def medianSorted(arr):
    return arr[len(arr)//2]
    

L = [random.randint(1,10) for _ in range(25)]
sL = sorted(L)
print("The random array:",L)
print("mean:",mean(L),", median:",medianSorted(sL),", mode:",modeSorted(sL))
