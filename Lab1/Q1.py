# Assuming order of elements in a pair doesn't matter
def checkPairs(L):
    count = 0
    for i in range(len(L)):
        for j in L[i+1:]: # Picks only numbers that haven't been used in the outer loop to avoid repetition
            if(L[i]+j==10):
                count+=1
    return count

arr = [2,7,4,1,3,6]
print("The number of pairs in the list",arr,"that equal to 10 is:",checkPairs(arr))
