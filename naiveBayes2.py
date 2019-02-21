import csv
import itertools
from multiprocessing import Pool
import multiprocessing
from math import log as ln
import pprint
import pandas as pd

NUM_CPUS = multiprocessing.cpu_count()

# Import vocabulary data
with open("20newsgroups/vocabulary.txt", "r") as f:
    vocab_data = f.readlines()
    vocabulary = []
    word_ID = 1
    for word in vocab_data:
        vocabulary.append ((word.strip(), word_ID))
        word_ID +=1

number_of_words_vocabulary = len(vocabulary)

#Import train label 
with open("20newsgroups/train_label.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    y_train = []
    document_ID = 1
    for category in readCSV:
        y_train.append((category[0], document_ID))
        document_ID += 1

# Import train data
with open("20newsgroups/train_data.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    X_train = []
    for item in readCSV:
        X_train.append((item[0], item[1], item[2]))

# Import test data
with open("20newsgroups/test_data.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    X_test = []
    for item in readCSV:
        X_test.append((item[0], item[1], item[2]))

#Import test label 
with open("20newsgroups/test_label.csv") as csvfile:
    readCSV = csv.reader(csvfile, delimiter=",")
    y_test = []
    document_ID = 1
    for category in readCSV:
        y_test.append((category[0], document_ID))
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

def dictOfCategoriesTest():
    dict_of_categories = {}
    for item in y_test:
        category = item[0]
        if category not in dict_of_categories:
            dict_of_categories[category] = 1
        else:
            dict_of_categories[category] +=1
    return(dict_of_categories)

def classPrior(c, dictCategoriesDocuments): # class prior probability
    docs_in_category = dictCategoriesDocuments[c]
    #print(docs_in_category)
    return(docs_in_category/number_of_train_documents)

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

def dictOfDocsPerCat():
    d ={} # dictionary, key=category, values=doc IDs
    for cat, document_ID in y_train:
        if cat in d:
            d[cat].add(str(document_ID))
        else:
            d[cat] = {str(document_ID)}
    return(d)

def numberOfTimesWordK_inCatC(category, dictCategoryDocumentIDs=None):
    d = {}
    if dictCategoryDocumentIDs is None:
        dictCategoryDocumentIDs = dictOfDocsPerCat()
    #print(category)
    docsInCat = dictCategoryDocumentIDs.get(str(category))
    #print(docsInCat)
    for doc_ID, word_ID, count in X_train:
        if doc_ID in docsInCat:
            if word_ID not in d:
                d[word_ID] = int(count)
            else:
                d[word_ID] += int(count)
        else:
            continue
    return(d)

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

def dictOfDocIDs_WordIDsTest(): # testing data, dictionary, key: document ID, values: word IDs
    d = {}
    for document_ID, word_ID, count in X_test:
        #document_ID = item[0]
        #word_ID = item[1]
        if document_ID in d:
            d[document_ID].append(word_ID)
        else:
            d[document_ID] = [word_ID]
    return(d)

def MLE(word_ID, category, dictCategoryWords, listWordIDNumberOfOccurences):
    #if dictWordIDNumberOfOccurences is None:
        #dictWordIDNumberOfOccurences = numberOfTimesWordK_inCatC(category, dictCategoryDocumentIDs)
    wordFreqinCategory = listWordIDNumberOfOccurences[int(category)-1]
    n_k = wordFreqinCategory.get(word_ID, 0)
    n = dictCategoryWords.get(str(category))
    P_MLE = n_k/n
    #print(word_ID, category, P_MLE)
    return (word_ID, category, P_MLE)

def BE(word_ID, category, dictCategoryWords, listWordIDNumberOfOccurences):
    #if dictWordIDNumberOfOccurences is None:
        #dictWordIDNumberOfOccurences = numberOfTimesWordK_inCatC(category, dictCategoryDocumentIDs)
    wordFreqinCategory = listWordIDNumberOfOccurences[int(category)-1]
    n_k = wordFreqinCategory.get(word_ID, 0)
    n = dictCategoryWords.get(str(category))
    P_BE = (n_k + 1)/(n + number_of_words_vocabulary)
    return(word_ID, category, P_BE)

def OmegaNB_MLE(document_ID, listOfCategories, dictCategoriesDocuments, listOfWords, dictCategoryWords, listWordIDNumberOfOccurences, dictDocumentIDWordIDs=None):
    if dictDocumentIDWordIDs is None:
        dictDocumentIDWordIDs = dictOfDocIDs_WordIDs()
    wordsInDocument_ID = dictDocumentIDWordIDs.get(str(document_ID))
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

def OmegaNB_BE(document_ID, listOfCategories, dictCategoriesDocuments, listOfWords, dictCategoryWords, listWordIDNumberOfOccurences, dictDocumentIDWordIDs=None):
    if dictDocumentIDWordIDs is None:
        dictDocumentIDWordIDs = dictOfDocIDs_WordIDs()
    wordsInDocument_ID = dictDocumentIDWordIDs.get(str(document_ID))
    probabilityPerCategory = {}
    for j in listOfCategories:
        ln_P_omega_j = ln(classPrior(j, dictCategoriesDocuments))
        ln_P_xi_given_omega_j = 0
        for i in wordsInDocument_ID:
            if i in listOfWords: # only words in vocaulary
                prob =  BE(i, j, dictCategoryWords, listWordIDNumberOfOccurences)[2]
                if prob == 0:
                    ln_P_xi_given_omega_j = ln_P_xi_given_omega_j + ln(1) # to avoid ln(0), which is undefined
                else:
                    ln_P_xi_given_omega_j = ln_P_xi_given_omega_j + ln(BE(i, j, dictCategoryWords, listWordIDNumberOfOccurences)[2])
            else:
                ln_P_xi_given_omega_j = ln_P_xi_given_omega_j + 0

        omega_NB = ln_P_omega_j + ln_P_xi_given_omega_j
        probabilityPerCategory[int(j)] = omega_NB
  
    result = max(probabilityPerCategory, key=probabilityPerCategory.get)
    return(result)

# Predict
def Predict_MLE(document_ID, listOfCategories, listOfWords, dictDocumentIDWordIDsTest, storedPriors, listPosteriors):
    if dictDocumentIDWordIDsTest is None:
        dictDocumentIDWordIDsTest = dictOfDocIDs_WordIDsTest()
    wordsInDocument_ID = dictDocumentIDWordIDsTest.get(str(document_ID))
    #print(wordsInDocument_ID)
    probabilityPerCategory = {}
    for j in listOfCategories:
        ln_P_omega_j = ln(storedPriors.get(int(j)))
        ln_P_xi_given_omega_j = 0
        for i in wordsInDocument_ID:
            if i in listOfWords: # only words in vocaulary
                prob =  (listPosteriors[int(j)-1]).get(i)
                if prob == 0:
                    ln_P_xi_given_omega_j = ln_P_xi_given_omega_j + ln(1) # to avoid ln(0), which is undefined
                else:
                    ln_P_xi_given_omega_j = ln_P_xi_given_omega_j + ln(prob)
            else:
                ln_P_xi_given_omega_j = ln_P_xi_given_omega_j + 0

        omega_NB = ln_P_omega_j + ln_P_xi_given_omega_j
        probabilityPerCategory[int(j)] = omega_NB
  
    result = min(probabilityPerCategory, key=probabilityPerCategory.get)
    #print(document_ID, result)
    return(result)
# Predict
def Predict_BE(document_ID, listOfCategories, listOfWords, dictDocumentIDWordIDsTest, storedPriors, listPosteriorsBE):
    if dictDocumentIDWordIDsTest is None:
        dictDocumentIDWordIDsTest = dictOfDocIDs_WordIDsTest()
    wordsInDocument_ID = dictDocumentIDWordIDsTest.get(str(document_ID))
    #print(wordsInDocument_ID)
    probabilityPerCategory = {}
    for j in listOfCategories:
        ln_P_omega_j = ln(storedPriors.get(int(j)))
        ln_P_xi_given_omega_j = 0
        for i in wordsInDocument_ID:
            if i in listOfWords: # only words in vocaulary
                prob =  (listPosteriorsBE[int(j)-1]).get(i)
                if prob == 0:
                    ln_P_xi_given_omega_j = ln_P_xi_given_omega_j + ln(1) # to avoid ln(0), which is undefined
                else:
                    ln_P_xi_given_omega_j = ln_P_xi_given_omega_j + ln(prob)
            else:
                ln_P_xi_given_omega_j = ln_P_xi_given_omega_j + 0

        omega_NB = ln_P_omega_j + ln_P_xi_given_omega_j
        probabilityPerCategory[int(j)] = omega_NB
  
    result = max(probabilityPerCategory, key=probabilityPerCategory.get)
    return(result)

def confusionMatrix(listOfTules):
    matrix = []
    for i in range(20):
        matrix.append([])
        for j in range(20):
            matrix[i].append(0)
    
    for actual, predicted in listOfTules:
        matrix[actual-1][predicted-1] +=1
    return(matrix)


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

    # train label in form of dictionary
    dictYtrain = {}
    for a, b in y_train:
        dictYtrain.setdefault(str(b), a)

    # Import test label
    with open("20newsgroups/test_label.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        y_test = []
        document_ID = 1
        for item in readCSV:
            y_test.append((item[0], document_ID))
            document_ID += 1

    # test label in form of dictionary
    dictYtest = {}
    for a, b in y_test:
        dictYtest.setdefault(str(b), a)

    # Import train data
    with open("20newsgroups/train_data.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        X_train = []
        for item in readCSV:
            X_train.append((item[0], item[1], item[2]))

    # Import test data
    with open("20newsgroups/test_data.csv") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        X_test= []
        for item in readCSV:
            X_test.append((item[0], item[1], item[2]))
    
    # list of all word IDs
    listOfWords = []
    for item in vocabulary:
        word_ID = item[1]
        listOfWords.append(str(word_ID))

    # list of all categories
    listOfCategories = []
    for i in range(1,21):
        listOfCategories.append(str(i))

    # total number of words in vocabulary
    number_of_words_vocabulary = len(vocabulary)

    # total number of training documents
    number_of_train_documents = len(y_train)

    # dictionary: key = category, values: number of documents
    dictCategoriesDocuments = dictOfCategories()
    dictCategoriesDocumentsTest = dictOfCategoriesTest()

    # dictionary: key=document ID, value=number of words
    dictDocumentsWords = dict_of_documents()

    # dictionary: key=category, value=number of words
    dictCategoryWords = wordsPerCategory(dictDocumentsWords)

    # dictionary, key=category, values=doc IDs
    dictCategoryDocumentIDs = dictOfDocsPerCat()

    # list of dictionaries, where list at 0 is cat 1, list at 1 is cat 2 etc., dictionary, key=wordID, number of occurences
    parallel = True
    iteritems = list(zip(range(1, 21), 20*[dictCategoryDocumentIDs]))
    num_cpus = multiprocessing.cpu_count()
    if parallel:
        print('\nMultitheaded computations are using', num_cpus, 'cpus')
        with Pool(num_cpus) as p:
            listWordIDNumberOfOccurences = list(p.starmap(numberOfTimesWordK_inCatC, iteritems))
    else:
        listWordIDNumberOfOccurences = list(itertools.starmap(numberOfTimesWordK_inCatC, iteritems))

    # dictionary, key: document ID, values: word IDs
    dictDocumentIDWordIDs = dictOfDocIDs_WordIDs()

    # dictionary, key: document ID, values: word IDs
    dictDocumentIDWordIDsTest = dictOfDocIDs_WordIDsTest()


    listOfDocumentIDs = set()
    for i in X_train:
        doc_ID = str(i[0])
        if doc_ID not in listOfDocumentIDs:
            listOfDocumentIDs.add(doc_ID)

    listOfDocumentIDsTest = set()
    for i in X_test:
        doc_ID = str(i[0])
        if doc_ID not in listOfDocumentIDsTest:
            listOfDocumentIDsTest.add(doc_ID)
    #print(listOfDocumentIDsTest)

    #Output class priors for 20 categories
    storedPriors = {}
    print("\nClass priors:")
    for i in range(1,21):
        storedPriors[i] = classPrior(str(i), dictCategoriesDocuments)
        print("P(Omega = ",i,")", classPrior(str(i), dictCategoriesDocuments))

    listPosteriors = [dict() for x in range(20)]
    for category in listOfCategories:
        for word in listOfWords:
            (word_ID, cat, P_MLE) = MLE(word, category, dictCategoryWords, listWordIDNumberOfOccurences)
            dcat = listPosteriors[int(category)-1]
            if word_ID not in dcat:
                dcat[word_ID] = P_MLE
            else:
                dcat[word_ID] += P_MLE

    listPosteriorsBE = [dict() for x in range(20)]
    for category in listOfCategories:
        for word in listOfWords:
            (word_ID, cat, P_BE) = BE(word, category, dictCategoryWords, listWordIDNumberOfOccurences)
            dcat = listPosteriorsBE[int(category)-1]
            if word_ID not in dcat:
                dcat[word_ID] = P_BE
            else:
                dcat[word_ID] += P_BE
############################################################################################
    # MLE classification (Training Data)
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
    listActualPredicted = []
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
        listActualPredicted.append((actualValue, predictedValue))


    # MLE Train Results
    print("\nOverall accuracy for MLE (Training)= ", numberOfCorrectlyClassifiedDocsTrainingData/len(listOfDocumentIDs))
    print("\nClass Accuracy for MLE (Training):")
    for i in listOfCategories:
        print("Group ", i, ":", (dictCategoryAccuracy.get(int(i), 0))/(dictCategoriesDocuments.get(i)))
    print("\nMLE Confusion Matrix (Training):\n")
    cols = ["[1]", "[2]", "[3]", "[4]", "[5]", "[6]", "[7]", "[8]", "[9]", "[10]", "[11]", "[12]", "[13]", "[14]", "[15]", "[16]", "[17]", "[18]", "[19]", "[20]"]
    df1 = pd.DataFrame(confusionMatrix(listActualPredicted), columns=cols, index=cols)
    print(df1.to_string())
###########################################################################################
    # BE Classification (Training data)
    def gen_OmegaNB_BE_args(listOfDocumentIDs,
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

    #return(trainClassification)
    with Pool(NUM_CPUS) as p:
        classificationsBE = p.starmap(OmegaNB_BE,
                                    gen_OmegaNB_BE_args(listOfDocumentIDs,
                                                         listOfCategories,
                                                         dictCategoriesDocuments,
                                                         set(listOfWords),
                                                         dictCategoryWords,
                                                         listWordIDNumberOfOccurences,
                                                         dictDocumentIDWordIDs))


    trainClassificationBE = {k: v for (k, v) in zip(listOfDocumentIDs, classificationsBE)}

    # Accuracy
    numberOfCorrectlyClassifiedDocsTrainingDataBE = 0
    dictCategoryAccuracyBE = {}
    listActualPredictedBE = []
    for key in trainClassificationBE:
        actualValue = int(dictYtrain.get(key))
        predictedValueBE = trainClassificationBE.get(key)
        if  actualValue == predictedValueBE:
            numberOfCorrectlyClassifiedDocsTrainingDataBE += 1
            if actualValue not in dictCategoryAccuracyBE:
                dictCategoryAccuracyBE[actualValue] = 1
            else:
                dictCategoryAccuracyBE[actualValue] += 1
        else:
            numberOfCorrectlyClassifiedDocsTrainingDataBE += 0
        listActualPredictedBE.append((actualValue, predictedValueBE))

    # BE Train Results
    print("\nOverall accuracy for BE (Training)= ", numberOfCorrectlyClassifiedDocsTrainingDataBE/len(listOfDocumentIDs))
    print("\nClass Accuracy for BE (Training):")
    for i in listOfCategories:
        print("Group ", i, ":", (dictCategoryAccuracyBE.get(int(i), 0))/(dictCategoriesDocuments.get(i)))
    print("\nBE Confusion Matrix (Training):\n")
    cols = ["[1]", "[2]", "[3]", "[4]", "[5]", "[6]", "[7]", "[8]", "[9]", "[10]", "[11]", "[12]", "[13]", "[14]", "[15]", "[16]", "[17]", "[18]", "[19]", "[20]"]
    df2 = pd.DataFrame(confusionMatrix(listActualPredictedBE), columns=cols, index=cols)
    print(df2.to_string())
##################################################################################
# MLE Classification (Testing data)
    def gen_Predict_MLE_args(listOfDocumentIDsTest, listOfCategories, listOfWords, dictDocumentIDWordIDsTest, storedPriors, listPosteriors):
        for doc_ID in listOfDocumentIDsTest:
            yield (doc_ID, listOfCategories, listOfWords, dictDocumentIDWordIDsTest, storedPriors, listPosteriors)

    
    with Pool(NUM_CPUS) as p:
        classificationsTestMLE = p.starmap(Predict_MLE,
                                    gen_Predict_MLE_args(listOfDocumentIDsTest, listOfCategories, set(listOfWords), dictDocumentIDWordIDsTest, storedPriors, listPosteriors))

    #print("check1")
    testMLEClassification = {k: v for (k, v) in zip(listOfDocumentIDsTest, classificationsTestMLE)}
    #print("check2")
    #print(testMLEClassification)

    # Accuracy
    numberOfCorrectlyClassifiedDocsTestinggDataMLE = 0
    dictCategoryAccuracyTestMLE = {}
    listActualPredictedTestMLE = []
    for key in testMLEClassification:
        actualValueTest = int(dictYtest.get(key))
        predictedValueTestMLE = testMLEClassification.get(key)
        if  actualValueTest == predictedValueTestMLE:
            numberOfCorrectlyClassifiedDocsTestinggDataMLE += 1
            if actualValueTest not in dictCategoryAccuracyTestMLE:
                dictCategoryAccuracyTestMLE[actualValueTest] = 1
            else:
                dictCategoryAccuracyTestMLE[actualValueTest] += 1

        else:
            numberOfCorrectlyClassifiedDocsTestinggDataMLE += 0
        listActualPredictedTestMLE.append((actualValueTest, predictedValueTestMLE))

    # MLE Test Results
    print("\nOverall accuracy for MLE (Testing Data) = ", numberOfCorrectlyClassifiedDocsTestinggDataMLE/len(listOfDocumentIDsTest))
    print("\nClass Accuracy for MLE (Testing Data):")
    for i in listOfCategories:
        print("Group ", i, ":", (dictCategoryAccuracyTestMLE.get(int(i), 0))/(dictCategoriesDocumentsTest.get(i)))
    print("\nMLE Confusion Matrix (Testing Data):\n")
    cols = ["[1]", "[2]", "[3]", "[4]", "[5]", "[6]", "[7]", "[8]", "[9]", "[10]", "[11]", "[12]", "[13]", "[14]", "[15]", "[16]", "[17]", "[18]", "[19]", "[20]"]
    df3 = pd.DataFrame(confusionMatrix(listActualPredictedTestMLE), columns=cols, index=cols)
    print(df3.to_string())

##################################################################################
# BE Classification (Testing data)

    def gen_Predict_BE_args(listOfDocumentIDsTest, listOfCategories, listOfWords, dictDocumentIDWordIDsTest, storedPriors, listPosteriorsBE):
        for doc_ID in listOfDocumentIDsTest:
            yield (doc_ID, listOfCategories, listOfWords, dictDocumentIDWordIDsTest, storedPriors, listPosteriorsBE)

    with Pool(NUM_CPUS) as p:
        classificationsTestBE = p.starmap(Predict_BE,
                                    gen_Predict_MLE_args(listOfDocumentIDsTest, listOfCategories, set(listOfWords), dictDocumentIDWordIDsTest, storedPriors, listPosteriorsBE))

    testBEClassification = {k: v for (k, v) in zip(listOfDocumentIDsTest, classificationsTestBE)}

    # Accuracy
    numberOfCorrectlyClassifiedDocsTestinggDataBE = 0
    dictCategoryAccuracyTestBE = {}
    listActualPredictedTestBE = []
    for key in testBEClassification:
        actualValueTest = int(dictYtest.get(key))
        predictedValueTestBE = testBEClassification.get(key)
        if  actualValueTest == predictedValueTestBE:
            numberOfCorrectlyClassifiedDocsTestinggDataBE += 1
            if actualValueTest not in dictCategoryAccuracyTestBE:
                dictCategoryAccuracyTestBE[actualValueTest] = 1
            else:
                dictCategoryAccuracyTestBE[actualValueTest] += 1
        else:
            numberOfCorrectlyClassifiedDocsTestinggDataBE += 0
        listActualPredictedTestBE.append((actualValueTest, predictedValueTestBE))

    # BE Test Results
    print("\nOverall accuracy for BE (Testing Data) = ", numberOfCorrectlyClassifiedDocsTestinggDataBE/len(listOfDocumentIDsTest))
    print("\nClass Accuracy for BE (Testing Data):")
    for i in listOfCategories:
        print("Group ", i, ":", (dictCategoryAccuracyTestBE.get(int(i), 0))/(dictCategoriesDocumentsTest.get(i)))
    print("\nBE Confusion Matrix (Testing Data):\n")
    cols = ["[1]", "[2]", "[3]", "[4]", "[5]", "[6]", "[7]", "[8]", "[9]", "[10]", "[11]", "[12]", "[13]", "[14]", "[15]", "[16]", "[17]", "[18]", "[19]", "[20]"]
    df4 = pd.DataFrame(confusionMatrix(listActualPredictedTestBE), columns=cols, index=cols)
    print(df4.to_string())

if __name__ == '__main__':
    main()
