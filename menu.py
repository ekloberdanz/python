from ellieMath import *

user_input = int(input ("press 1 for factorial\n" 
                    "press 2 for prime numbers\n"
                    "press 3 for mean\n"
                    "press 4 for combination\n"
                    "press 5 for variance\n"))

if user_input == 1:
    number_input = input ("enter a number")
    result = factorial (number_input)
    print ("The factorial of", number_input, "is", result)
    