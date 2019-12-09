#!/usr/bin/env python
# coding: utf-8

# # Initial Setup and Load Data

# In[1]:


# Load Libraries
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

import re
import string
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 


# In[2]:


# For maximizing Cell width of Jupyter Notebook Cells
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[3]:


# Set working directory
os.chdir("/opt/ShadData/NLP/Directi_CaseStudy/")


# In[4]:


# For Not Displaying warnings
import warnings
warnings.filterwarnings('ignore')


# #### Problem Statement​ -
# Go Daddy is a domain reseller/retailer for Radix TLDs. Radix utilizes domain lifetime value metric for all strategic and tactical decisions related to the business. The first step of determining the domain lifetime value is to predict the renewal rate of a domain. Given the data we get from Go Daddy you need to come up with a machine learning model to predict the renewal rate of the domains registered.

# In[5]:


# Load data
df_train = pd.read_csv("gd_assignment_train_data.csv")
df_test = pd.read_csv("gd_assignment_test_data.csv")


# In[6]:


# Display Shape of dataframe (Rows, Columns)
df_train.shape , df_test.shape


# In[7]:


# Display Top 5 rows of dataframe
df_train.head()


# In[8]:


# Check Datatypes of all the columns present in the dataset
df_train.dtypes


# The details of data attributes in the dataset are as follows:
# 
# - domain: [STRING] Registered domain
# - creation_date: [DATE] Rate of registration of domain
# - expiry_date: [DATE] Date when the domain expires/comes up for renewal
# - reseller: [STRING] Go Daddy (reseller where the domain was registered)
# - registrant_country: [STRING/FACTOR] Country of customer who registered the domain
# - reg_price: [NUMERIC] Price at which the domain was registered (USD)
# - renewal_status: [FACTOR/BINARY] Whether the domain renew or did not renew before the expiry_date

# In[9]:


# Describe Columns of Dataframe and check mean, std, quartiles, min-max for all variables
df_train.describe()


# In[10]:


df_train.columns


# # Exploratory Data Analysis

# In[11]:


###################################################### Exploratory Data Analysis ####################################################

################################ Converting to appropriate datatypes as per Problem Statement

# 1) Converting expiry_date/creation_date to datetime format for both train and test dataset
# 2) Converting renewal_status and registrant_country as category
# 3) Removing index variable 'unnamed' since it won't contribute towards about analysis

complete_data = [df_train, df_test]

for data in complete_data:
    data['expiry_date']  = pd.to_datetime(data['expiry_date'],errors='coerce')
    data['creation_date']  = pd.to_datetime(data['creation_date'],errors='coerce')
    data['renewal_status']= data['renewal_status'].astype('category',ordered=False)
    data['registrant_country']= data['registrant_country'].astype('category',ordered=False)

df_train = df_train.drop(['Unnamed: 0'], axis=1)
df_test = df_test.drop(['Unnamed: 0'], axis=1)


# In[12]:


### Checking datatypes of train data
df_train.dtypes


# In[13]:


### Checking datatypes of test data
df_test.dtypes


# In[14]:


# Check Number of Unique values present in each variable
df_train.nunique()


# In[15]:


# Unique values present in reseller variable on Train Data
df_train.reseller.value_counts()


# In[16]:


print("Unique Values Present in Reseller : {}".format(df_train.reseller.unique()[0]))


# In[17]:


# Unique values present in reseller variable on Test Data
df_test.reseller.value_counts()


# In[18]:


print("Unique Values Present in Reseller : {}".format(df_test.reseller.unique()[0]))


# In[19]:


# 4) Removing Reseller variable from both datasets since It has only one category ['go daddy'] which won't help us in analysis
df_train = df_train.drop(['reseller'], axis=1)
df_test = df_test.drop(['reseller'], axis=1)


# In[20]:


# 5) Check values of Target Variables and convert to 1's (Renewed) and 0's (Not Renewed)
df_train.renewal_status.value_counts()


# In[21]:


df_train['renewal_status'] = df_train['renewal_status'].replace(["Not Renewd"], 0)
df_train['renewal_status'] = df_train['renewal_status'].replace(["Renewed"], 1)

df_test['renewal_status'] = df_test['renewal_status'].replace(["Not Renewd"], 0)
df_test['renewal_status'] = df_test['renewal_status'].replace(["Renewed"], 1)


# # Missing Value Analysis

# In[22]:


################ Missing Value Analysis for Train Data ###############

#Creating dataframe with number of missing values
missing_val = pd.DataFrame(df_train.isnull().sum())

#Reset the index to get row names as columns
missing_val = missing_val.reset_index()

#Rename the columns
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})

#Calculate percentage
missing_val['Missing_percentage'] = (missing_val['Missing_percentage']/len(df_train))*100

#Sort the rows according to decreasing missing percentage
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)

#Save output to csv file
missing_val.to_csv("Missing_percentage.csv", index = False)

missing_val


# In[23]:


################ Missing Value Analysis for Test Data ###############

#Creating dataframe with number of missing values
missing_val = pd.DataFrame(df_train.isnull().sum())

#Reset the index to get row names as columns
missing_val = missing_val.reset_index()

#Rename the columns
missing_val = missing_val.rename(columns = {'index': 'Variables', 0: 'Missing_percentage'})

#Calculate percentage
missing_val['Missing_percentage'] = (missing_val['Missing_percentage']/len(df_train))*100

#Sort the rows according to decreasing missing percentage
missing_val = missing_val.sort_values('Missing_percentage', ascending = False).reset_index(drop = True)

#Save output to csv file
missing_val.to_csv("Missing_percentage.csv", index = False)

missing_val


# # Domain Name Analysis and Feature Engineering

# In[24]:


################################### Domain Name Analysis ######################################

#Count of domains with '.online' extension
df_train.domain.str.find(".online").count()


# In[25]:


# Replace .online with empty since its the only domain
df_train['domain'] = df_train['domain'].apply(lambda x: re.sub(r'\b.online\b', '', x))


# In[26]:


df_test['domain'] = df_test['domain'].apply(lambda x: re.sub(r'\b.online\b', '', x))


# In[27]:


# Wordcloud for TOP Domain names found in our train data

from wordcloud import WordCloud
plt.figure(figsize = (12, 8))
text = ' '.join(df_train.domain.values)
wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top words in Domains')
plt.axis("off")
plt.show()


# In[28]:


# # wordcloud for Cases with Renewal Status as 1 (Renewed)
renewal_status_1 = df_train.loc[df_train['renewal_status'] == 1]

plt.figure(figsize = (12, 8))
text = ' '.join(renewal_status_1['domain'].values)
wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top words in Domain with Renewal Status 1')
plt.axis("off")
plt.show()


# In[29]:


# # wordcloud for Cases with Renewal Status as 0 (Not Renewed)
renewal_status_0 = df_train.loc[df_train['renewal_status'] == 0]

plt.figure(figsize = (12, 8))
text = ' '.join(renewal_status_0['domain'].values)
wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1000).generate(text)
plt.imshow(wordcloud)
plt.title('Top words in Domain with Renewal Status 0')
plt.axis("off")
plt.show()


# In[30]:


### Function to Find whether a word is Gibberish or Valid Word without dictionary lookup
from __future__ import division
import re
import math


def split_in_chunks(text, chunk_size):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i + chunk_size])
    if len(chunks) > 1 and len(chunks[-1]) < 10:
        chunks[-2] += chunks[-1]
        chunks.pop(-1)
    return chunks


def unique_chars_per_chunk_percentage(text, chunk_size):
    chunks = split_in_chunks(text, chunk_size)
    unique_chars_percentages = []
    for chunk in chunks:
        total = len(chunk)
        unique = len(set(chunk))
        unique_chars_percentages.append(unique / total)
    return sum(unique_chars_percentages) / len(unique_chars_percentages) * 100


def vowels_percentage(text):
    vowels = 0
    total = 0
    for c in text:
        if not c.isalpha():
            continue
        total += 1
        if c in "aeiouAEIOU":
            vowels += 1
    if total != 0:
        return vowels / total * 100
    else:
        return 0


def word_to_char_ratio(text):
    chars = len(text)
    words = len([x for x in re.split(r"[\W_]", text) if x.strip() != ""])
    return words / chars * 100


def deviation_score(percentage, lower_bound, upper_bound):
    if percentage < lower_bound:
        return math.log(lower_bound - percentage, lower_bound) * 100
    elif percentage > upper_bound:
        return math.log(percentage - upper_bound, 100 - upper_bound) * 100
    else:
        return 0


def classify(text):
    if text is None or len(text) == 0:
        return 0.0
    ucpcp = unique_chars_per_chunk_percentage(text, 35)
    vp = vowels_percentage(text)
    wtcr = word_to_char_ratio(text)

    ucpcp_dev = max(deviation_score(ucpcp, 45, 50), 1)
    vp_dev = max(deviation_score(vp, 35, 45), 1)
    wtcr_dev = max(deviation_score(wtcr, 15, 20), 1)

    return max((math.log10(ucpcp_dev) + math.log10(vp_dev) +
                math.log10(wtcr_dev)) / 6 * 100, 1)


# In[31]:


df_train['GibberishOrNot'] = df_train.domain.apply(classify)
df_test['GibberishOrNot'] = df_test.domain.apply(classify)


# In[32]:


### Feature Engineer new variables from Domain Names 
complete_data = [df_train, df_test]

for data in complete_data:
    ## Number of words in the Domain ##
    data["Domain_num_words"] = data["domain"].apply(lambda x: len(str(x)))

    ## Number of Digits in the Domain ##
    data["Domain_num_words_digit"] = data["domain"].apply(lambda x: len([w for w in str(x) if w.isdigit()]))

    ## Number of Alphabetical Words in the Domain ##
    data["Domain_num_words_aplha"] = data["domain"].apply(lambda x: len([w for w in str(x) if w.isalpha()]))

    ## Number of Identifier words in the Domain ##
    data["Domain_num_words_identifier"] = data["domain"].apply(lambda x: len([w for w in str(x) if w.isidentifier()]))
    
    ## Number of Alphanumeric words in the Domain ##
    data["Domain_num_words_alnum"] = data["domain"].apply(lambda x: len([w for w in str(x) if w.isalnum()]))
    
    ## Number of characters in the Domain ##
    data["Domain_num_chars"] = data["domain"].apply(lambda x: len(str(x)))

    ## Number of stopwords in the Domain ##
    data["Domain_num_stopwords"] = data["domain"].apply(lambda x: len([w for w in str(x).lower() if w in stop_words]))

    ## Number of punctuations in the Domain ##
    data["Domain_num_punctuations"] =data['domain'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))


# In[33]:


df_train.columns


# In[34]:


df_train.head()


# # Date Columns - Feature Engineering

# In[35]:


################### Date conversion - Feature Engineering ###################
def seasons(x):
    ''' for seasons in a year using month column'''
    if (x >=3) and (x <= 5):
        return 'spring'
    elif (x >=6) and (x <=8 ):
        return 'summer'
    elif (x >= 9) and (x <= 11):
        return'fall'
    elif (x >=12)|(x <= 2) :
        return 'winter'

def week(x):
    ''' for week:weekday/weekend in a day_of_week column '''
    if (x >=0) and (x <= 4):
        return 'weekday'
    elif (x >=5) and (x <=6 ):
        return 'weekend'


# In[36]:


## Deriving new values like year, month, day_of_the_week, season, week from date
complete_data = [df_train, df_test]
for data in complete_data:
    for col in ['expiry_date', 'creation_date']:
        data[col+"_year"] = data[col].apply(lambda row: row.year)
        data[col+"_month"] = data[col].apply(lambda row: row.month)
        data[col+"_day_of_week"] = data[col].apply(lambda row: row.dayofweek)
        data[col+'_seasons'] = data[col+"_month"].apply(seasons)
        data[col+'_week'] = data[col+"_day_of_week"].apply(week)


# In[37]:


# Create new variable which takes into account the difference between expiry date and creation Date
df_train['Diff_Days_Expiry_Creation'] = df_train['expiry_date'] - df_train['creation_date']
df_train['Diff_Days_Expiry_Creation'] = df_train['Diff_Days_Expiry_Creation']/np.timedelta64(1,'D')


# In[38]:


# Create new variable which takes into account the difference between expiry date and creation Date
df_test['Diff_Days_Expiry_Creation'] = df_test['expiry_date'] - df_train['creation_date']
df_test['Diff_Days_Expiry_Creation'] = df_test['Diff_Days_Expiry_Creation']/np.timedelta64(1,'D')


# In[39]:


categorical_columns = ['registrant_country', 'expiry_date_year','expiry_date_month', 'expiry_date_day_of_week', 'expiry_date_seasons',
       'expiry_date_week', 'creation_date_year', 'creation_date_month','creation_date_day_of_week', 'creation_date_seasons','creation_date_week']

#numerical_columns = [x for x in df_train.columns if x not in categorical_columns]

numerical_columns = ['reg_price',
                     'GibberishOrNot',
                     'Domain_num_words',
                     'Domain_num_words_digit',
                     'Domain_num_words_aplha',
                     'Domain_num_words_identifier',
                     'Domain_num_words_alnum',
                     'Domain_num_chars',
                     'Domain_num_stopwords',
                     'Domain_num_punctuations',
                     'Diff_Days_Expiry_Creation']


# # Graphical Visualization of Data

# In[40]:


############## Graphical Exploratory Data Analysis ##############
plt.figure(figsize=(20,10))
sns.set(font_scale = 2)
sns.countplot(df_train.renewal_status)


# In[41]:


# Frequency Countplot for Categorical variable
for i in categorical_columns:
    if i == "registrant_country":
        continue
    sns.factorplot(data=df_train, x=i ,kind='count', size=6, aspect=2)


# In[42]:


# Bivariate Analysis of all categorical variables with Target Variable.
for col in categorical_columns:
    if col == "registrant_country":
        continue
    df_cat = pd.DataFrame(df_train.groupby([col], as_index=False).sum())
    sns.catplot(x=col, y="reg_price", data=df_cat.reset_index(), kind="point", height=6, aspect=2)


# In[43]:


# Histogram - Distribution plot for all continous variables
for i,col in enumerate(numerical_columns):
    plt.figure(i)
    plt.axvline(df_train[col].mean(), 0,0.7, color = 'g')
    sns.distplot(df_train[col], color = 'b')


# # Statistical Tests for Checking Correlation and Dependence

# In[44]:


# Create df from continous variables
df_corr = df_train.loc[:,numerical_columns]


# In[45]:


# Corelation graph for checking Multicollinearity for continous variables
f, ax = plt.subplots(figsize=(15, 10))

#Generate correlation matrix
corr = df_corr.corr()

#Plot heatmap using seaborn library
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
ax = sns.heatmap(corr, mask=mask, vmax=.1, square=True, annot=False, ax=ax)

# We will have l2 regularization for automatically removing all the highly correlated features


# In[46]:


#loop for ANOVA test for checking dependancy of categorical and numerical data Type
for i in numerical_columns:
    f, p = stats.f_oneway(df_train.renewal_status, df_train[i])
    print("P value for variable "+str(i)+" is "+str(p))


# # GLoVe - Word Embedding for Domain Feature

# In[47]:


# load the GloVe vectors in a dictionary:
from tqdm import tqdm
embeddings_index = {}
f = open('glove.840B.300d.txt', encoding='utf8')
for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))


# In[48]:


# this function creates a normalized vector for the whole sentence
def word2vec(s):
    words = str(s).lower()
    words = [w for w in words if w.isalpha()]
    words = [w for w in words if w not in string.punctuation]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())


# In[49]:


xtrain_glove = [word2vec(x) for x in tqdm(df_train['domain'].values)]
xtest_glove = [word2vec(x) for x in tqdm(df_test['domain'].values)]


# In[50]:


xtrain_glove = pd.DataFrame(np.array(xtrain_glove))
xtest_glove = pd.DataFrame(np.array(xtest_glove))

print(xtrain_glove.shape)


# In[51]:


xtrain_glove.columns = ['glove_syn_'+str(i) for i in range(300)]
xtest_glove.columns = ['glove_syn_'+str(i) for i in range(300)]

df_train = pd.concat([df_train, xtrain_glove], axis=1)
df_test = pd.concat([df_test, xtest_glove], axis=1)


# In[52]:


df_train.columns


# In[53]:


# Drop Date Columns and domain since we've extracted all possible information from them 
df_train = df_train.drop(['domain', 'expiry_date', 'creation_date'], axis = 1)
df_test =  df_test.drop(['domain','expiry_date','creation_date'], axis=1)


# In[54]:


df_train.shape


# In[55]:


df_train.dtypes


# # One Hot Encoding For Categorical Data

# In[56]:


# One hot Encoding
temp = pd.get_dummies(pd.concat([df_train, df_test], keys=[0,1]), columns=categorical_columns)

# # Selecting data from multi index and assigning them i.e
df_train,df_test = temp.xs(0),temp.xs(1)


# In[57]:


df_train.columns


# # Model Development

# In[58]:


######################## Model Building #####################
X_train = df_train.iloc[:, df_train.columns != 'renewal_status'].values
y_train = df_train['renewal_status'].values

# import the ML algorithm
from sklearn.linear_model import LogisticRegression

# Instantiate the classifier
LogReg = LogisticRegression(penalty='l2', C=1.0)

# Train classifier
LogReg.fit(X_train, y_train)


# In[59]:


# Do Prediction
y_pred = LogReg.predict(df_test.iloc[:, df_test.columns != 'renewal_status'].values)


# In[60]:


###################################### Evaluation Metrics ###############################################
y = df_test['renewal_status'].values

print("Confusion Matrix : ",end='\n')
print(confusion_matrix(y, y_pred))
   
print('Accuracy : ', accuracy_score(y, y_pred), end='\n')
    
print('Classification Report : ', end='\n')
print(classification_report(y, y_pred))


# # Building Model with only Important Features

# In[61]:


important_features = [
 'reg_price',
 'GibberishOrNot',
 'Domain_num_words_digit',
 'Domain_num_chars',
 'Domain_num_punctuations',
 'renewal_status'
]


# In[62]:


df_train_new = df_train.loc[:, important_features]
df_test_new = df_test.loc[:, important_features]


# In[63]:


######################## Model Building #####################
X_train = df_train_new.iloc[:, df_train_new.columns != 'renewal_status'].values
y_train = df_train_new['renewal_status'].values

# import the ML algorithm
from sklearn.linear_model import LogisticRegression

# Instantiate the classifier
LogReg = LogisticRegression(penalty='l2', C=1.0)

# Train classifier
LogReg.fit(X_train, y_train)


# In[64]:


# Do Prediction
y_pred = LogReg.predict(df_test_new.iloc[:, df_test_new.columns != 'renewal_status'].values)


# In[65]:


###################################### Evaluation Metrics ###############################################
y = df_test_new['renewal_status'].values

print("Confusion Matrix : ",end='\n')
print(confusion_matrix(y, y_pred))

print()
   
print('Accuracy : ', accuracy_score(y, y_pred) * 100, end='\n')

print()

print('Classification Report : ', end='\n')
print(classification_report(y, y_pred))


# # Saving Model 

# In[66]:


# Pickling of Final Classification Model
import pickle
pickle.dump(LogReg, open('final_prediction.pickle', 'wb'))


# In[67]:


#### Conclsuion : 

#1) Built a model to predict Domain Renewal with almost 94% Testing Accuracy
#2) Explored through Exploratory Data Analysis and understood the relationship of features.
#3) Built Wordcloud to better understand what are the TOP MOST Domain Name words.
#4) Used statistical analysis (ANOVA, correlation-coefficients) and graphs.
#5) Did alot of Feaure Engineering and built new features to assist the model to better understand the patterns inside the data:
    #1) Dervied new features Like Year, Month, Season from Variables including Date (creation_date, expiry_date)
    #2) Reshaped the entire dataset from 8 columns to 580 columns with same rows.
    #3) Applied Textual Preprocessing and cleaning techniques to create features from Domain Names.
    #4) Using NLP - GLoVe Word Embeddings Transformed domain names to respective Vector Representations

