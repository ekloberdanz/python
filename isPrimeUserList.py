import IsPrime

def str_to_int(l):
    k = []
    for i in l:
        i = int(i)
        k.append(i)
    return k

def IsPrimeUserList():
    user_input = input("Enter a list of numbers: ")
    l = user_input.split(" ")
    l = str_to_int(l)
    k = []
    for i in l:
        if IsPrime.IsPrime(i):
            k.append(i)
    return k


#print(str_to_int([1, 2, 3]))

#print(IsPrimeUserList())