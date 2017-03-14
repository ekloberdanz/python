from Stack import Stack

def Hist():
    user_input = ""
    a = []
    b = []
    c = []
    d = []
    while user_input!= "Done":
        user_input = input("Enter a number between 0 and 100 per line: \n" )
        if user_input.isdigit():

            item = int(user_input)
            if item in range (0,25):
                a.append(item)
            elif item in range (25,50):
                b.append(item)
            elif item in range (50,75):
                c.append(item)
            else:
                d.append(item)
    return a,b,c,d

a,b,c,d = Hist()
print('[0,25]', len(a) * '*')
print('[25,50]', len(b) * '*')
print('[50,75]', len(c) * '*')
print('[75, 100]', len(d) * '*')



