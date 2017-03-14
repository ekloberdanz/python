
def IsPrime_3to100(self):
    for self in range (3, 100):
        divisor = 2
        while divisor < num[0:len(num)]:
            if num%divisor == 0:
                return False
            else:
                divisor = divisor + 1
        return True




print(IsPrime_3to100())