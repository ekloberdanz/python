'''
def factorial(n):
    result = n
    i = 1
    while i < n:
        result = result * (n - i)
        i = i + 1
    return result

print(factorial(3))
print(factorial(9))

'''

def GCD(a,b):
    if a == b:
        return a
    if a < b:
        return GCD(a, b - a)
    if a > b:
        return GCD(a - b, b)


print(GCD(32, 6))