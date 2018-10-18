# Importing all the necessary libraries
import os
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
import string
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import random
from collections import OrderedDict
nltk.download('punkt')




# Set the workding directory and import the file
os.chdir(r'path to working directory')
df = pd.read_excel(r'sheet name having the data')



# Doing the house working for text clean up and normalization
porter = PorterStemmer()
table = str.maketrans('', '', string.punctuation)
stop_words = set(stopwords.words('english'))


# Function to do the text clean up
def texnormalize(text, table, stop_words):
    tokens = word_tokenize(str(text))
    tokens = [w.lower() for w in tokens]
    words = [w.translate(table) for w in tokens]
    words = [w for w in words if not w in stop_words]
    words = [porter.stem(word) for word in words]
    words = [word for word in words if len(word) > 0]
    string = ' '.join(words)
    return string
    

# Vectoring the function for faster processing
vtexnormalize = np.vectorize(texnormalize, otypes=[object])



# Creating a copy of the dataframe for further processing keeping the original one untouched
dt = df.copy()



# Specify the column names from which values to be fetched for output preparation
output_cols = ['col_1', 'col_2', 'col3_3']



# Move all the output columns to extreme right of the dataframe
cols = dt.columns.tolist()
for col in output_cols:
    n = int(cols.index(col))
    cols = cols[:n] + cols[n+1:] + [cols[n]]

dt = dt[cols]



# Cleaning the text across all the input columns (to be treated as input criteria) of the dataframe. 
col_list = list(dt.columns)
criteria_cols = dt.shape[0] - len(output_cols)
for i in col_list[0:criteria_cols]:
    if dt[i].dtypes == 'object':
        dt[i] = vtexnormalize(dt[i].values, table, stop_words)



# Cleaning the column names as well
dt.columns = [texnormalize(col, table, stop_words) for col in list(dt.columns)]



# House keeping the original and changed column values
orig_cols = list(df.columns)
new_cols = list(dt.columns)



# Creating a dict with key as unique value in the dataframe and value as the corresponding column name for the input columns only
dict_unique_val = {}
for i in list(dt.columns)[0:criteria_cols]:
    for j in dt[i].unique():
        if j not in dict_unique_val.keys():
            dict_unique_val[str(j)] = [str(i)]
        else:
            dict_unique_val[str(j)].append(str(i))




# Sorting the dict based on the length of the keys. This will help in finding intended columns
dict_unique_val = dict(OrderedDict(sorted(dict_unique_val.items(), key = lambda t: len(t[0]), reverse=True)))




# Creating a single string for all the unique values in the dataframe excluding the column names and values from the target(output) columns
all_val_string_set = ''
for i in dict_unique_val.keys():
    i = str(i)
    all_val_string_set = all_val_string_set + ' ' + i



# From the user input find the column names and corresponding values for data processing (basically filtering to get the required output)

def get_entity_values(user_input):
    entities_val = {}
    user_input = texnormalize(user_input, table, stop_words)
    user_input_copy = " " + user_input + " "
    for i in dict_unique_val.keys():
        entity_hint = ''
        wrapped_i = " " + str(i) + " "                          # Searching for a bag of words in stead of just string search
        if wrapped_i in user_input_copy:
            user_input_copy = user_input_copy.replace(i, '')    # Avoiding fetching of other unnecessary columns based on matching string
            if len(dict_unique_val[i]) > 1:
                entity_hint_given = 0
                entity_highest_priority = 0
                for j in range(len(dict_unique_val[i])):
                    entity_priority = 0
                    for k in dict_unique_val[i][j].split():     # split of jth col name for row value of i
                        if k in user_input.split():
                            entity_hint_given = 1
                            entity_priority += 1                # Finds how many hint words of jth col name present in input
                    if entity_priority > entity_highest_priority:
                        entity_highest_priority = entity_priority
                        entity_hint = dict_unique_val[i][j]     # Assigns the col name for which maximum word hints given
                if entity_hint_given == 0:                      # "This will assign 1st col if multiple columns are found
                    entity_hint = dict_unique_val[i][0]         # for given row value but no hint given for any column name"
            else:
                entity_hint = dict_unique_val[i][0]             # If only one col is found for given row value then just assign that
            entities_val[entity_hint] = str(i)                  # Map the col name with the row value in a new dictionary
    return entities_val
                        
                        


col_list = list(dt.columns)




# Specify keys words that the bot would expect in the input to treat them as output column names. 
# Example 1: "pric" for output column name "MRP Price"
# Example 2: "quantit" for output column name "Quantity Bought"
def get_intent_values(user_input):
    intents = []
    if 'key_word_1' in texnormalize(user_input, table, stop_words):
        response_intent = 'output_col_1'
        intents.append(response_intent)
    if 'key_word_1' in texnormalize(user_input, table, stop_words):
        response_intent = 'output_col_1'
        intents.append(response_intent)
    if 'key_word_1' in texnormalize(user_input, table, stop_words):
        response_intent = 'output_col_1'
        intents.append(response_intent)
    if 'key_word_2' in texnormalize(user_input, table, stop_words):
        response_intent = 'output_col_2'
        intents.append(response_intent)
    if 'key_word_2' in texnormalize(user_input, table, stop_words):
        response_intent = 'output_col_2'
        intents.append(response_intent)
    if 'key_word_3' in texnormalize(user_input, table, stop_words):
        response_intent = 'output_col_3'
        intents.append(response_intent)
    if 'key_word_3' in texnormalize(user_input, table, stop_words):
        response_intent = 'output_col_3'     
        intents.append(response_intent)
    intents = list(set(intents))
    return intents




# Dictionary to create a mapping for changed names of target or output columns with original ones for reporting only
dict_intent_present = {'output_col_1' : 'output_col_1_actual_name',
                       'output_col_2' : 'output_col_2_actual_name',
                       'output_col_3' : 'output_col_3_actual_name'}



# Get the response from the bot for the given input
def get_bot_response(user_input, dict_intent_present):
    filter_condition = get_entity_values(user_input)
    asked_data = get_intent_values(user_input)
    response = []
    incorrect_input = ''
    for i in asked_data:
        df_temp = dt.copy()
        for j, k in filter_condition.items():
            if df_temp[str(j)].dtypes == 'int64':
                k = int(k)
            elif df_temp[str(j)].dtypes == 'float64':
                k = float(k)
            else:
                k = str(k)
            df_temp = df_temp[df_temp[str(j)] == k]
            if df_temp.shape[0] == 0:
                incorrect_input = j
                break
        response.append(str(dict_intent_present[i]) + ' : ' + str(df_temp[i].sum()))
    result_row_count = df_temp.shape[0]
    return response, filter_condition, result_row_count, incorrect_input




# Hardcoded values to handle the greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["Hi", "Hey", "Hi there", "Hello", "I am glad! You are talking to me"]

def greeting(user_input):
     for word in user_input.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Dictionary of changed and original column names
dict_cols = {}
for i in range(len(new_cols)):
    dict_cols[new_cols[i]] = orig_cols[i]


# Dictionary of changed and original values in the dataframe
dict_vals = {}
for i in range(dt.shape[0]):
    for j in range(dt.shape[1]):
        dict_vals[str(dt.iloc[i, j])] = str(df.iloc[i, j])



# The code to start an active conversation between user and the bot
flag=True
print("\nChatbot: I am a Bot. I will answer to your queries. If you want to exit, type Bye!")

while(flag==True):
    user_input = input("Me: ")
    user_input = user_input.lower()
    if(user_input!='bye'):
        if(user_input == 'thanks' or user_input == 'thank you'):
            flag = False
            print("\nChatbot: You are welcome!")
        else:
            if (greeting(user_input) != None):
                print("\nChatbot: " + greeting(user_input))
            else:
                bot_response_val, filter_condition, result_row_count, incorrect_input = get_bot_response(user_input, dict_intent_present)
                print("\nChatbot: Here is the received input criteria", end = '\n')
                for i in filter_condition.keys():
                    print(dict_cols[i], '=>', dict_vals[filter_condition[i]])
                if result_row_count == 0:
                    print("\nChatbot: There is no result found for the given input. Seems like incorrect value given for - ", dict_cols[incorrect_input], ' Please check and enter again.', end = '\n')
                else:
                    print("\nChatbot: Total number of rows found for the given input: ", result_row_count)
                    print("\nChatbot: Here is the summed up result ", end = "\n")
                    for i in bot_response_val:
                        print(i)
    else:
        flag=False
        print("\nChatbot: Bye! take care.")



