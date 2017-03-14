def factorial (x):
    if x == 1 or x== 0:
        return 1
    else:
        return x * (factorial(x-1)) 
        
    
def isPrime(x):
    for y in range (2,x):
        if x%y==0:
            return (False)
    return (True)

def mean (L):
    total=0
    i=0
    while i < len(L):
        total=L[i]+total
        i = i + 1
    return total/len(L)