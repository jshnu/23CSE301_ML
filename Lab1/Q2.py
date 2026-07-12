def rangeL(arr):
    if len(arr)<3:
        return -1

    mini=maxi=arr[0]
    for i in range(1,len(arr)):
        if arr[i]<mini:
            mini=arr[i]
        elif arr[i]>maxi:
            maxi=arr[i]
    return (maxi-mini)

L= list(eval(input("Enter a list of numbers: ")))
x = rangeL(L)
if (x!=-1):
    print("Range of the list is:",x)
else:
    print("Range determination not possible")
