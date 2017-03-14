import operator
def clean(s):
    word = ""
    for letter in s:
        if letter.isalpha():
            word = word+letter
            #print(word)
    return word.lower()

input_file = open("a-study-in-scarlet.txt", "r")

count_dict = {}
for line in input_file:
    #print(line, end="")
    split_line = line.split(" ")
    #print(split_line)
    for word in split_line:
        word = clean(word)
        if not word:
            continue
        
        if word in count_dict:
            count_dict[word] = count_dict[word] + 1
        else:
            count_dict[word] = 1
#print(count_dict)
#print(count_dict["spending"])

sorted_count = sorted(count_dict.items(), key=operator.itemgetter(1))
sorted_count.reverse()
#print(sorted_count[:20])
output_file = open("top-words.csv", "w")
for word, count in sorted_count:
    output_file.write(word + "," + str(count) + "\n")