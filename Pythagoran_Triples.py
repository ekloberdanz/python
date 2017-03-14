def PT():
    user_input = input("Enter a number: ")
    user_input = int(user_input)
    l = []
    for item in range(0,user_input):
        x = item
        y = item and y > x
        z = item and z > y

        if x**2 + y**2 == z**2:
            l.append(x)
            l.append(y)
            l.append(z)
        return l

print(PT())
