def BinarySearch (L,target):
    #middle=L[len(L)/2]
    low=0
    high=len(L)-1
    while True:
        middle_index = (high+low)//2        
        middle_element = L[middle_index]
        if low>high:
            return (-1)
        elif target < middle_element:            
            high = middle_index - 1
        elif target > middle_element:
            low = middle_index + 1
        elif target == middle_element:
            return middle_index
        

        
            