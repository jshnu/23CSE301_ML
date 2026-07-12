def maxCount(s):
    d = dict()
    for i in s:
        if i not in d:
            d[i]=1
        else:
            d[i]+=1
    ans = (s[0],d[s[0]])
    for j in d:
        if d[j]>ans[1]:
            ans = (j,d[j])
    return ans

str1 = input("Enter a string:")
ans = maxCount(str1)
print(f"The maximum occurring character is '{ans[0]}' and it's count is {ans[1]}.")
