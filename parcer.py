#input_file = open("/home/eliska/Downloads/a-study-in-scarlet.txt", "r")


def removePunctuation(s):
    ret_s = ""
    for letter in s:
        if letter.isalpha():
            ret_s+= letter
    return ret_s

'''
def parser():
    #for line in input_file:
    line = "A study in Scarlet is a Doyle's novel."
    line = line.lower()
    line = line.split()
    sentence = []
    for letter in line:
        line = removePunctuation(letter)
        sentence = sentence.append(line)


    return sentence
'''

'''
def parser():
    user_input = input("Enter text: ")
    l = []
    user_input = user_input.lower()
    list_of_words = user_input.split()
    for item in  list_of_words:
        item = removePunctuation(item)
        l.append(item)

    return l


print(parser())

'''
from encryption import encrypt
def parser(text):
    l = []
    text = text.lower()
    list_of_words = text.split()
    for item in  list_of_words:
        item = removePunctuation(item)
        l.append(item)

    return l

def kitty(text):
    # 1) parse text, i.e. call parser
    text = parser(text)

    # 2) encrypt the text
    text_as_string = ""
    for i in text:
        i = str(i)
        text_as_string = text_as_string + i

    # 3) return the encrypted text
    text_encrypted = encrypt(text_as_string, 1)
    return text_encrypted



text = input("Hello Kitty!")
cipher_text = kitty(text)
print(cipher_text)







