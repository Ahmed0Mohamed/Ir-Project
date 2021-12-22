'''**************************************** import liprary *********************************************'''
from nltk.util import pr
import numpy as np
import os
import nltk
nltk.download('stopwords')
from nltk.stem import lancaster
from nltk.tokenize import TweetTokenizer
from natsort import natsorted
from nltk.corpus import stopwords
import math
from tabulate import tabulate
import pandas as pd
stop_words = set(stopwords.words('english'))
import string


def read_file(filename):
    with open(filename, 'r', encoding="utf-8", errors="surrogateescape") as f:
        stuff = f.read()

    f.close()

    return stuff


'''******************************** Preprocessing all terms in files ******************************'''
def preprocessing(final_string):
    '''***************** Tokenize ************************'''

    tokenizer = TweetTokenizer()
    token_list = tokenizer.tokenize(final_string)

    '''***************** Remove punctuations ************************'''
    table = str.maketrans('', '', '\t')
    token_list = [word.translate(table) for word in token_list]
    punctuations = (string.punctuation).replace("'", "")
    trans_table = str.maketrans('', '', punctuations)
    stripped_words = [word.translate(trans_table) for word in token_list]
    token_list = [str for str in stripped_words if str]

    '''*********** Change to lowercase ************************'''
    token_list = [word.lower() for word in token_list]
    freqForEachFolder = len(token_list)  
    return token_list,freqForEachFolder

    '''***************** create folder of files that will br in positional index ************************'''


folder_names = ["ir"]

# Initialize the stemmer.
stemmer = lancaster.LancasterStemmer()

# Initialize the file no.
fileno = 0

# Initialize the dictionary.
pos_index = {}

# Initialize the file mapping (fileno -> file name).
file_map = {}

#freq for each file 
freq = {} 
counter = 0
alltokens = []
for folder_name in folder_names:

    # Open files.
    file_names = natsorted(os.listdir( folder_name))

    # For every file.
    for file_name in file_names:

        # Read file contents.
        stuff = read_file( folder_name + "/" + file_name)

        # This is the list of words in order of the text.
        # We need to preserve the order because we require positions.
        # 'preprocessing' function does some basic punctuation removal,
        # stopword removal etc.
        final_token_list,freq[counter] = preprocessing(stuff) 
        # For position and term in the tokens.
        for pos, term in enumerate(final_token_list):

            # First stem the term.
            term = stemmer.stem(term)
            #print(term)
            # If term already exists in the positional index dictionary.
            if term in pos_index:

                # Increment total freq by 1.
                pos_index[term][0] = pos_index[term][0] + 1

                # Check if the term has existed in that DocID before.
                if fileno in pos_index[term][1]:
                    pos_index[term][1][fileno].append(pos)
                    

                else:
                    pos_index[term][1][fileno] = [pos]

                # If term does not exist in the positional index dictionary
            # (first encounter).
            else:

                # Initialize the list.
                pos_index[term] = []
                # The total frequency is 1.
                pos_index[term].append(1)
                # The postings list is initially empty.
                pos_index[term].append({})
                # Add doc ID to postings list.
                pos_index[term][1][fileno] = [pos]
                alltokens.append(term)

            # Map the file no. to the file name.
        file_map[fileno] = folder_name + "/" + file_name
        # Increment the file no. counter for document ID mapping
        fileno += 1
        counter +=1
print(pos_index)
'''**************************************** Query *********************************************'''

pharseQuery=input("Enter your pharse Query To search for it :")

'''********************** stopWord and filter it with stemmer on query *****************************'''

stop_words = set(stopwords.words('english'))
pharseQuery, freq2 = preprocessing(pharseQuery)


Filtered_pharseQuery = [w for w in pharseQuery if not w in stop_words]

stemmer = lancaster.LancasterStemmer()

'''************************* Sample positional index to test the code *****************************'''

x=len(Filtered_pharseQuery)
i=0
arr = []
for i in range(0,x):
    Filtered_pharseQuery[i] = stemmer.stem(Filtered_pharseQuery[i])
    try:
        sample_pos_idx = pos_index[Filtered_pharseQuery[i]]
        print("\nPositional Index --> "+str(i+1)+" word")
        #print(sample_pos_idx)
        frequancey=sample_pos_idx[0]
        print("Frequancey :",frequancey)


        file_list = sample_pos_idx[1]
        print("\nFilename, [Positions]")
        counter=0
        for fileno, positions in file_list.items():
            counter+= len(positions)
            print(file_map[fileno], positions)
       
        print("df :",counter)
        df=counter
        idf=math.log10(10/df)
        print("IDF :",idf)

    except KeyError:
        print("\nSorry!Your "+str(i+1)+" word"+" Not Found" )

normalizate = [[0 for c in range(len(file_names))]for r in range(len(alltokens))]
doc_len =[0 for r in range(len(file_names))]
idf = []
tf_idf = [[0 for c in range(len(file_names))]for r in range(len(alltokens))]
tf_weights = [[0 for c in range(len(file_names))]for r in range(len(alltokens))]
word = 0
display = [[0 for c in range(len(file_names))]for r in range(len(alltokens))]
for term in alltokens:
    try:
        index_word= alltokens.index(term) 
        sample_pos_idx = pos_index[term]
        file_list = sample_pos_idx[1]
        counter=0
        FFreq = 0
        for fileno, positions in file_list.items():
            FFreq=len(positions)
            counter += FFreq 
            tf_weights[index_word][fileno]= 1 + math.log10(FFreq) 
            display[index_word][fileno]=1

        df=counter
        idf.append(math.log10(10/df))
        word +=1
    except KeyError:
        print("\nSorry!Your "+str(i+1)+" word"+" Not Found" )

for i in range(len(file_names)):
    for j in range(len(idf)):
        tf_idf[j][i] = tf_weights[j][i]*idf[j]
        doc_len[i] += tf_idf[j][i]*tf_idf[j][i] 
    doc_len[i] = math.sqrt(doc_len[i])
    
for i in range(len(file_names)):
    for j in range(len(idf)):
        normalizate[j][i] = tf_idf[j][i]/doc_len[i] 
         

print("\ndisplay TF table\n")
print('\n'.join([''.join(['{:4}'.format(item) for item in row]) 
      for row in display ]))

print("\nprint idf table\n")
print(idf)

print("\nprint TF-weights table\n")
print('\n'.join([''.join(['{:15}'.format(item) for item in row]) 
      for row in tf_weights ]))

print("\nprint tf-idf table\n")
print('\n'.join([''.join(['{:}\t'.format(item) for item in row]) 
      for row in tf_idf ]))

print("\nprint doc-len\n")
print(doc_len)

print("\nprint normalization table\n")
print('\n'.join([''.join(['{:}\t'.format(item) for item in row]) 
      for row in normalizate ]))