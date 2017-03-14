def bubbble_sort(l):
    notsorted = True
    while notsorted == True:
        notsorted = False        
        for curr in range (0, len(l)-1):
            next = curr + 1
            if l[curr] > l[next]:

                temp = l[curr]
                l[curr] = l[next]
                l[next] = temp
                notsorted = True
              
    return l

l = [2, 5, 7, 8, 3, 2, 9, 1111]
print(bubbble_sort(l))