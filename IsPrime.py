'''
def IsPrime(num):
    if num < 2:
        return False
    divisor = 2
    while divisor < num:
        if num % divisor == 0:
            return False
        else:
            divisor = divisor + 1
    return True
'''
import math

def IsPrime(num):
    if num < 2:
        return False
    divisor = 2
    temp = int(math.sqrt(num)) + 1
    while divisor < temp:
        if num % divisor == 0:
            return False
        else:
            divisor = divisor + 1
    return True

'''
print(IsPrime(5))

for n in range (3,101):
    if IsPrime(n):
        print(n, "is prime")
    else:
        print(n, "is not prime")
'''
def primesInRange(a,b):
    l = []
    for n in range(a,b):
        if IsPrime(n):
            l.append(n)
    return l
'''
print(primesInRange(1,10))
print(primesInRange(1,103))
print(primesInRange(-100,100))
'''
def ListPrime(n):
    l = []
    x = 2
    while len(l) < n:
        if IsPrime(x):
            l.append(x)
        x = x + 1
    return l

"""
print(ListPrime(10))

print(IsPrime(7917))
"""