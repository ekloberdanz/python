from isPrimeUserList import str_to_int
def BinToDec():
    user_input = input("Enter a bimary number: ")
    exponent = 0
    result = 0
    for i in range(len(user_input) - 1, -1, -1):
        sub_result = int(user_input[i]) * 2**exponent
        result = result + sub_result
        exponent = exponent + 1
    return result
print(BinToDec())