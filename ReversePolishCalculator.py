from isPrimeUserList import str_to_int
from Stack import Stack

def evaluate(operator, val1, val2):
    if operator == '+':
        return val2 + val1
    elif operator == '-':
        return val2 - val1
    elif operator == '*':
        return val2 * val1
    elif operator == '/':
        return val2 / val1
    elif operator == '%':
        return val2%val1
    else:
        print('error, not an operator')

def RPC():
    user_input = input("Enter numbers followed by operators: ")
    user_input = user_input.split(" ")
    s = Stack()
    operators = ['+', '-', '*','/', '%']
    for item in user_input:
        if item not in operators:
            s.push(item)
        else:
            val1 = int(s.pop())
            val2 = int(s.pop())
            result = evaluate(item, val1, val2)
            s.push(result)
    return s.top()

i = 0
while True:

    print(RPC())
    i += 1

