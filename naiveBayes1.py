import csv
import itertools
from multiprocessing import Pool
import multiprocessing
from math import log as ln

NUM_CPUS = multiprocessing.cpu_count()

# Import vocabulary data
with open("20newsgroups/vocabulary.txt", "r") as f:
    vocab_data = f.readlines()
    vocabulary = []
    word_ID = 1
    for word in vocab_data:
        vocabulary.append ((word.strip(), word_ID))
        word_ID +=1

# length of individual words in vocabulary
#for word in vocabulary:
    #print(len(word[0]))

number_of_words_vocabulary = len(vocabulary)
#print(number_of_words_vocabulary)
#print(vocabulary)

#Import train label 
with open("20newsgroups/train_label.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    y_train = []
    document_ID = 1
    for category in readCSV:
        y_train.append((category[0], document_ID))
        document_ID += 1

number_of_train_documents = len(y_train)
#print(y_train)

#number of documents per category
def dictOfCategories():
    dict_of_categories = {}
    for item in y_train:
        category = item[0]
        if category not in dict_of_categories:
            dict_of_categories[category] = 1
        else:
            dict_of_categories[category] +=1
    return(dict_of_categories)
#dict_of_categories = sorted(dict_of_categories.items(), key=lambda x:x[0])
#print("sorted dict", dict_of_categories)
#print("categories", dict_of_categories)

def classPrior(c, dictCategoriesDocuments): # class prior probability
    docs_in_category = dictCategoriesDocuments[c]
    #print(docs_in_category)
    return(docs_in_category/number_of_train_documents)
'''
#Output class priors
for i in range(1,21):
    print("P(Omega = ",i,")", classPrior(str(i), dictCategoriesDocuments))
'''

#print(dict_of_categories)
#def numberOfWordsInCategory(c):

# Import train data
with open("20newsgroups/train_data.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    X_train = []
    for item in readCSV:
        X_train.append((item[0], item[1], item[2]))

s = 0
for i in X_train:
    s+=int((i[2]))

#print("sum of words", s)
#print(X_train)
#print((X_train[2])[2])

# dictionary: key=document ID, value=number of words
def dict_of_documents():
    d = {} # total number of words per document id
    for i in X_train:
        document = str(i[0])
        word = i[1]
        count = i[2]
        if document not in d:
            d[document] = int(count)
        else:
            d[document] += int(count)
    return(d)
#print("dict of docs", dict_of_documents)


#def numberOfWordsInDocument():
#for i in X_train[1:10]:
    #print (i[0],i[2])

def wordsPerCategory(dictDocumentsWords=None):
    n = {} # total number of words per category (n)
    for category, document_ID in y_train:
        if dictDocumentsWords is None:
            dictDocumentsWords = dict_of_documents()
        num_of_words = dictDocumentsWords.get(str(document_ID), 0)
        if category in n:
            n[category] += num_of_words
        else:
            n[category] = num_of_words
    return(n)
#print("word count", n)
#print(sum(n.values()))
#print(wordsPerCategory())
def dictOfDocsPerCat():
    d ={} # dictionary, key=category, values=doc IDs
    for cat, document_ID in y_train:
        if cat in d:
            d[cat].add(str(document_ID))
        else:
            d[cat] = {str(document_ID)}
    return(d)
#print(dictOfDocsPerCat.get('10'))

def numberOfTimesWordK_inCatC(category, dictCategoryDocumentIDs=None):
    d = {}
    if dictCategoryDocumentIDs is None:
        dictCategoryDocumentIDs = dictOfDocsPerCat()
    #print(category)
    docsInCat = dictCategoryDocumentIDs.get(str(category))
    #print(docsInCat)
    for doc_ID, word_ID, count in X_train:
        if doc_ID in docsInCat:
        #for document_ID in dictOfDocsPerCat.get(str(document_ID), 0):
            if word_ID not in d:
                d[word_ID] = int(count)
            else:
                d[word_ID] += int(count)
        else:
            continue
    return(d)


#print(numberOfTimesWordK_inCatC('10').get('39968'))


'''
for word in listOfWords:
    for category in listOfCategories:
        print("\nMaximum Likelihood estimator of word ID",word,"given category",category, "is" ,MLE(word, category))
        print("Bayesian estimator of word ID",word,"given category",category, "is", BE(word, category))
'''
def dictOfDocIDs_WordIDs(): # dictionary, key: document ID, values: word IDs
    d = {}
    for document_ID, word_ID, count in X_train:
        #document_ID = item[0]
        #word_ID = item[1]
        if document_ID in d:
            d[document_ID].append(word_ID)
        else:
            d[document_ID] = [word_ID]
    return(d)

#print(dictOfDocIDs_WordIDs())

def MLE(word_ID, category, dictCategoryWords, listWordIDNumberOfOccurences):
    #if dictWordIDNumberOfOccurences is None:
        #dictWordIDNumberOfOccurences = numberOfTimesWordK_inCatC(category, dictCategoryDocumentIDs)
    wordFreqinCategory = listWordIDNumberOfOccurences[int(category)-1]
    n_k = wordFreqinCategory.get(word_ID, 0)
    #print(n_k)
    n = dictCategoryWords.get(str(category))
    P_MLE = n_k/n
    return (word_ID, category, P_MLE)

def BE(word_ID, category, dictCategoryWords, listWordIDNumberOfOccurences):
    #if dictWordIDNumberOfOccurences is None:
        #dictWordIDNumberOfOccurences = numberOfTimesWordK_inCatC(category, dictCategoryDocumentIDs)
    wordFreqinCategory = listWordIDNumberOfOccurences[category-1]
    n_k = wordFreqinCategory.get(word_ID, 0)
    #print(n_k)
    n = dictCategoryWords.get(str(category))
    #print(n)
    P_BE = (n_k + 1)/(n + number_of_words_vocabulary)
    return(word_ID, category, P_BE)

def OmegaNB_MLE(document_ID, listOfCategories, dictCategoriesDocuments, listOfWords, dictCategoryWords, listWordIDNumberOfOccurences, dictDocumentIDWordIDs=None):
    if dictDocumentIDWordIDs is None:
        dictDocumentIDWordIDs = dictOfDocIDs_WordIDs()
    wordsInDocument_ID = dictDocumentIDWordIDs.get(str(document_ID))
    #if document_ID == '1':
        #print(document_ID, len(wordsInDocument_ID), wordsInDocument_ID)
    probabilityPerCategory = {}
    for j in listOfCategories:
        ln_P_omega_j = ln(classPrior(j, dictCategoriesDocuments))
        ln_P_xi_given_omega_j = 0
        for i in wordsInDocument_ID:
            if i in listOfWords: # only words in vocaulary
                prob =  MLE(i, j, dictCategoryWords, listWordIDNumberOfOccurences)[2]
                if prob == 0:
                    ln_P_xi_given_omega_j = ln_P_xi_given_omega_j + ln(1) # to avoid ln(0), which is undefined
                else:
                    ln_P_xi_given_omega_j = ln_P_xi_given_omega_j + ln(MLE(i, j, dictCategoryWords, listWordIDNumberOfOccurences)[2])
            else:
                ln_P_xi_given_omega_j = ln_P_xi_given_omega_j + 0

        omega_NB = ln_P_omega_j + ln_P_xi_given_omega_j
        probabilityPerCategory[int(j)] = omega_NB
  
    result = min(probabilityPerCategory, key=probabilityPerCategory.get)
    return(result)

def OmegaNB_BE(document_ID, dictDocumentIDWordIDs=None):
    if dictDocumentIDWordIDs is None:
        dictDocumentIDWordIDs = dictOfDocIDs_WordIDs()
    wordsInDocument_ID = dictDocumentIDWordIDs.get(document_ID)

    probabilityPerCategory = []
    for j in listOfCategories:
        ln_P_omega_j = ln(classPrior(j))

        ln_P_xi_given_omega_j = 0
        for i in wordsInDocument_ID:
            if i in listOfWords: # only words in vocaulary
                ln_P_xi_given_omega_j += ln(BE(i, j))

        omega_NB = ln_P_omega_j + ln_P_xi_given_omega_j
        probabilityPerCategory.append(omega_NB)
    return(max(probabilityPerCategory)) 


def main():
    # Import vocabulary data
    with open("20newsgroups/vocabulary.txt", "r") as f:
        vocab_data = f.readlines()
        vocabulary = []
        word_ID = 1
        for word in vocab_data:
            vocabulary.append ((word.strip(), word_ID))
            word_ID +=1

    #Import train label 
    with open("20newsgroups/train_label.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        y_train = []
        document_ID = 1
        for item in readCSV:
            y_train.append((item[0], document_ID))
            document_ID += 1

    dictYtrain = {}
    for a, b in y_train:
        dictYtrain.setdefault(str(b), a)
    l=[]
    for i in range (1,700):
        l.append(str(i))


    # Import train data
    with open("20newsgroups/train_data.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        X_train = []
        for item in readCSV:
            X_train.append((item[0], item[1], item[2]))

    # list of all word IDs
    listOfWords = []
    for item in vocabulary:
        word_ID = item[1]
        listOfWords.append(str(word_ID))

    # list of all categories
    listOfCategories = []
    for i in range(1,21):
        listOfCategories.append(str(i))
    #print(listOfCategories)
    # total number of words in vocabulary
    number_of_words_vocabulary = len(vocabulary)

    # total number of training documents
    number_of_train_documents = len(y_train)

    # dictionary: key = category, values: number of documents
    dictCategoriesDocuments = dictOfCategories()

    #Output class priors for 20 categories
    for i in range(1,21):
        print("P(Omega = ",i,")", classPrior(str(i), dictCategoriesDocuments))

    # dictionary: key=document ID, value=number of words
    dictDocumentsWords = dict_of_documents()

    # dictionary: key=category, value=number of words
    dictCategoryWords = wordsPerCategory(dictDocumentsWords)

    # dictionary, key=category, values=doc IDs
    dictCategoryDocumentIDs = dictOfDocsPerCat()
    #print(dictCategoryDocumentIDs)
    
    # list of dictionaries, where list at 0 is cat 1, list at 1 is cat 2 etc., dictionary, key=wordID, number of occurences
    parallel = True
    iteritems = list(zip(range(1, 21), 20*[dictCategoryDocumentIDs]))
    num_cpus = multiprocessing.cpu_count()
    if parallel:
        print('using', num_cpus, 'cpus')
        with Pool(num_cpus) as p:
            listWordIDNumberOfOccurences = list(p.starmap(numberOfTimesWordK_inCatC, iteritems))
    else:
        listWordIDNumberOfOccurences = list(itertools.starmap(numberOfTimesWordK_inCatC, iteritems))
    #print('testing list', listWordIDNumberOfOccurences)

    # dictionary, key: document ID, values: word IDs
    dictDocumentIDWordIDs = dictOfDocIDs_WordIDs()

    '''
    # MLE, list of dictionaries, each dict represents a category, key=word ID, values: probability of word
    parallel = True

    len_iteritems = len(listOfWords) * 20
    categories_gen = [cat for _ in listOfWords for cat in range(1, 21)]
    iteritems = list(zip([word_id for word_id in listOfWords for _ in range(1,21)],
                    categories_gen,
                    [dictCategoryWords for _ in range(len_iteritems)],
                    [listWordIDNumberOfOccurences for _ in range(len_iteritems)]))

    num_cpus = multiprocessing.cpu_count()
    # MLE
    if parallel:
        print('using', num_cpus, 'cpus')
        with Pool(num_cpus) as p:
            listWordMLEs = list(p.starmap(MLE, iteritems)))
    else: 
        listWordMLEs = list(itertools.starmap(MLE, iteritems))

    #print('MLE\n',listWordMLEs)

    listCategoryWordsMLE = 20*[None]
    #import pdb
    #pdb.set_trace()
    for word_ID, category, P_MLE in listWordMLEs:
        #print(word_ID, category)
        if listCategoryWordsMLE[category-1] is None:
            listCategoryWordsMLE[category-1] = {word_ID: P_MLE}
        else:
            listCategoryWordsMLE[category-1][word_ID] = P_MLE

    '''
    # MLE classification

    listOfDocumentIDs = set()
    for i in X_train:
        doc_ID = str(i[0])
        if doc_ID not in listOfDocumentIDs:
            listOfDocumentIDs.add(doc_ID)
    #print(len(listOfDocumentIDs))
    #quit()


    def gen_OmegaNB_MLE_args(listOfDocumentIDs,
                             listOfCategories,
                             dictCategoriesDocuments,
                             listOfWords,
                             dictCategoryWords,
                             listWordIDNumberOfOccurences,
                             dictDocumentIDWordIDs):
        for doc_ID in listOfDocumentIDs:
            yield (doc_ID,
                   listOfCategories,
                   dictCategoriesDocuments,
                   listOfWords,
                   dictCategoryWords,
                   listWordIDNumberOfOccurences,
                   dictDocumentIDWordIDs)

    '''
    trainClassification = {}
    for doc_ID in listOfDocumentIDs:
        trainClassification[doc_ID] = OmegaNB_MLE(doc_ID,
                                                  listOfCategories,
                                                  dictCategoriesDocuments,
                                                  listOfWords,
                                                  dictCategoryWords,
                                                  listWordIDNumberOfOccurences,
                                                  dictDocumentIDWordIDs)'''
    #return(trainClassification)
    with Pool(NUM_CPUS) as p:
        classifications = p.starmap(OmegaNB_MLE,
                                    gen_OmegaNB_MLE_args(listOfDocumentIDs,
                                                         listOfCategories,
                                                         dictCategoriesDocuments,
                                                         set(listOfWords),
                                                         dictCategoryWords,
                                                         listWordIDNumberOfOccurences,
                                                         dictDocumentIDWordIDs))


    trainClassification = {k: v for (k, v) in zip(listOfDocumentIDs, classifications)}

    # Accuracy
    numberOfCorrectlyClassifiedDocsTrainingData = 0
    dictCategoryAccuracy = {}
    listOfMismatch = []
    for key in trainClassification:
        actualValue = int(dictYtrain.get(key))
        predictedValue = trainClassification.get(key)
        if  actualValue == predictedValue:
            numberOfCorrectlyClassifiedDocsTrainingData += 1
            if actualValue not in dictCategoryAccuracy:
                dictCategoryAccuracy[actualValue] = 1
            else:
                dictCategoryAccuracy[actualValue] += 1

        else:
            numberOfCorrectlyClassifiedDocsTrainingData += 0
            listOfMismatch.append((actualValue, predictedValue))

    print(len(listOfMismatch))

    d = {}
    for item in listOfMismatch:
        if item[0] not in d:
            d[item[0]] = [item[1]]
        else:
            d[item[0]].append(item[1])
    print(listOfMismatch)
    print("MLE Confusion Matrix")
    for i in range(1,21):
        listMismatch_i = d.get(i)
        for i in listOfMismatch:
            di = {}
            a = trainClassification.get(i)
            if a not in di:
                di[a] = 1
            else:
                di[a] += 1
            #print(di)




    #trainConfusionMatrix = {}
   # for key in dictYtrain:
       # aValue = int(dictYtrain.get(key))
        #pValue = trainClassification.get(key)
       # if aValue not in trainConfusionMatrix:
            #trainConfusionMatrix[aValue] = 


    #print(dictCategoryAccuracy)
    print(numberOfCorrectlyClassifiedDocsTrainingData)
    print("Overall accuracy for MLE = ", numberOfCorrectlyClassifiedDocsTrainingData/len(listOfDocumentIDs))
    print("Class Accuracy for MLE:")
    for i in listOfCategories:
        print("Group ", i, ":", (dictCategoryAccuracy.get(int(i), 0))/(dictCategoriesDocuments.get(i)))

if __name__ == '__main__':
    main()
