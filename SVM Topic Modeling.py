#!/usr/bin/env python
# coding: utf-8

# ## Using the SVM Topic Module

# In[7]:


import os
import pandas as pd
import svm_topic


# In[8]:


os.getcwd()


# In[9]:


# read in the sample data: a sample of facebook posts from sites shared by members of congress in Labs report
sample_posts = pd.read_csv("facebook_posts_sample.csv", encoding = 'utf-8', sep = '\t')


# In[10]:


len(sample_posts)


# In[11]:


sample_posts.columns


# ### Defining an SVM topic object

# ######  This is a set of entities known to produce 'hard news', or stories that have a broad impact on Americans. Somewhat subjective.

# In[12]:


#pos_entities = list(pd.read_csv("hard_news-Copy1.txt", header = None)[0])
pos_entities = [u'Univision', u'MSNBC', u'NPR',
       u'RollCall', u'The Hill', u'Daily Caller',
       u'Washington Post', u'Weekly Standard', u'National Review',
       u'AP News', u'Hot Air', u'CNN', u'Independent Journal Review',
       u'Foreign Policy', u'The Atlantic', u'Vox', u'ABC News',
       u'RedState', u'CS Monitor', u'Splinter', u'Vice',
       u'Bloomberg', u'NBC News', 
       u'Rare America', u'Politico', u'PBS Newshour', u'Breitbart',
       u'Business Insider', u'HuffPost', u'Yahoo News',
       u'Washington Times',
       u'PBS', u'The Federalist', u'Axios', u'CNBC',
       u'Daily Kos', u'Newsweek', 
       u'US News', u'Washington Examiner', 
       u'Federal News Radio', u'The Daily Beast', u'New York Times',
       u'ThinkProgress', u'Reason', u'LA Times', u'Slate', u'Lifenews',
       u'Time', u'CBS News', u'Lifezette', u'Buzzfeed', 
       u'Townhall', u'TalkingPointsMemo', u'C-Span',
       u'New Republic', u'Mother Jones', u'USA Today',
       u'Salon', u'The Blaze',
       u'Conservative Review', u'McClatchy',
       u'National Journal', u'Newsmax', u'Gallup', u'Forbes',
       u'Free Beacon', u'Observer', 
       u'The Nation', u'MSN', 
       u'The Intercept', u'Politifact', u'GovExec',
       u'Daily Signal', u'Fox News', u'Fox Business', 
       u'Real Clear Politics']


# In[13]:


# Any text containing one of these fragments is considered 'political', and labeled positively if it came from hard news source 
pos_regex = 'politi|usnews|world|national|state|elect|vote|govern|campaign| war |polic|econ|unemploy|racis|energy|abortion|educa|healthcare|immigration'
# Any text containing one of these fragments is considered NOT political
neg_regex = 'ncaa|sports|entertainment|arts|fashion|style|lifestyle|leisure|celeb|movie|music|gossip|food|travel|horoscope|weather|gadget'


# In[14]:


# Create a new topic object, could be labeled 'politicaltopic' or somethign if you wanted 
# This assumes all of your text is in a specific column
newtopic = svm_topic.Svmtopic(sample_posts, text_column = 'textcol',
                    entities_column = 'publisher_name',
                    positive_regex = pos_regex, 
                    negative_regex = neg_regex, 
                    positive_entities = pos_entities, 
                    strip_src_text = False)


# ### Define the positive set and examine

# In[15]:


newtopic.define_positive() # uses the regex to define positive rows based on entities and regex
newtopic.create_binary_labels() # creates the actual labels that will be used in training


# In[16]:


# Examine a sample of positively labeled cases 
for line in newtopic.data[newtopic.data.positive_label==True].sample(5).textcol:
    print(line+ "\n")


# In[18]:


for line in newtopic.data[newtopic.data.negative_label==True].sample(5).textcol:
    print(line + "\n")


# ### Create the training data

# In[19]:


newtopic.create_training_data() # removes rows that fall into both positive and negative categories or neither


# ### Create the model

# In[20]:


newtopic.create_model() # actually runs a TF-IDF, then the SVM model itself on the training data


# ### Show informative model features

# In[21]:


newtopic.show_most_informative_features()


# ### Cross-validate the model and return CV accuracy

# In[22]:


newtopic.cross_validate()


# ### Show the 'delta', how many new posts are found by the model

# In[23]:


# What proportion of the original data were labeled as positive?
newtopic.label_based_proportion


# In[24]:


newtopic.model_based_increase


# In[27]:


# Show the total
newtopic.label_based_proportion + newtopic.model_based_increase


# In[26]:


len(newtopic.data)


# In[ ]:


# Number of newly-labeled political posts, not from source set
len(newtopic.data[(newtopic.data.predicted_labels==True) & (newtopic.data.positive_label==False)])


# In[28]:


# Print out a sample of the text that was 'found' by the model to be political, but didn't contain our seed set
for line in newtopic.data[(newtopic.data.predicted_labels==True) & (newtopic.data.positive_label==False)].sample(10).textcol:
    print(line + "\n")

