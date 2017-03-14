def encrypt(input_string, n):
    s = ""
    for letter in input_string:
        s = s + chr(ord(letter)+n)
    return s
