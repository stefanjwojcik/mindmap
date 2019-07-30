#!/usr/bin/env python
# coding: utf-8

# ## What is this topic analysis and how does it differ from topic modeling?

# ### Classic topic-modeling is built on the `bag of words’ approach, which clusters words that co-occur in documents under the assumption that they fall into conceptual ‘topics’
# * Very useful for talking about the ‘big picture’
# * Shows useful word associations that tell a story
# ### But it is also problematic: 
# * Can be slow on large, messy corpses
# * Can fail to identify a specific topic of interest, e.g. immigration
# * Can be hard to decide level of generality
# * Often involves lots of manual `tuning’
# 

# ### This technique is an approach to extract a predetermined topic or set of topics in a corpus
# 
# This is how it works. We want to isolate a specific topic of interest, but we don't know ALL the words people might use to describe it. For example, perhaps we are mining a database of academic research, and we want to find all discussions about the early earth. By this I mean the history of the earth as it progressed from a hot soup to a place sustaining various forms of life. We don't know all the ways this topic may have been described, but we do know SOME of the words that are almost certainly associated with it. 
# 
# We start assembling a set of words you think are very likely in the topic area - words for which it would be hard to imagine containing those words but not being in the topic area. For example, it's relatively hard to imagine the noun 'late cretaceous' occurring in a phrase that is not about the early earth, so we include it. For that matter, we can also include a variety of other technical references to time periods relating to the earth's early history (e.g. `pleistocene`). 
# 
# While we're doing so, we may know that there are words that are not relevant to earth's history, but may nonetheless  be likely to occur with our topic. In this case, we want to exclude words like this. For example, the words `paleontology` or `archaeology` might be in this group, since those words reference fields involved with excavation of artifacts in the earth rather than the early earth itself. We also might want to include popular dinosaurs in this list, like T-rex, to avoid costumes and 
# 
# But we know that it's more than just words - we know there are some academic journals that talk a lot about this topic, so we can also collect them in a list. 
# 

# In[ ]:


#* Known positive words in the topic (whitelisting words)
#* Known positive entities (whitelisting entities)
#* Known negative words (blacklisting words)
#* Known negative entities (blacklisting entities)


# In[ ]:


### What 


# In[1]:


import os
import pandas as pd
import svm_topic


# In[2]:


os.getcwd()


# In[3]:


# read in the sample data: a sample of facebook posts from sites shared by members of congress in Labs report
sample_posts = pd.read_csv("facebook_posts_sample.csv", encoding = 'utf-8', sep = '\t')


# In[4]:


len(sample_posts)


# In[5]:


sample_posts.columns


# ### Defining an SVM topic object

# ######  This is a set of entities known to produce 'hard news', or stories that have a broad impact on Americans. Somewhat subjective.

# In[6]:


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


# In[7]:


# Any text containing one of these fragments is considered 'political', and labeled positively if it came from hard news source 
pos_regex = 'weather|politi|usnews|world|national|state|elect|vote|govern|campaign| war |polic|econ|unemploy|racis|energy|abortion|educa|healthcare|immigration'
# Any text containing one of these fragments is considered NOT political
neg_regex = 'ncaa|sports|entertainment|arts|fashion|style|lifestyle|leisure|celeb|movie|music|gossip|food|travel|horoscope|gadget'


# In[8]:


pos_regex_k = pos_regex.split("|")


# In[9]:


# Create a new topic object, could be labeled 'politicaltopic' or somethign if you wanted 
# This assumes all of your text is in a specific column
newtopic = svm_topic.Svmtopic(sample_posts, text_column = 'textcol',
                    entities_column = 'publisher_name',
                    positive_regex = pos_regex, 
                    negative_regex = neg_regex, 
                    positive_entities = pos_entities, 
                    strip_src_text = False)


# ### Define the positive set and examine

# In[10]:


newtopic.define_positive() # uses the regex to define positive rows based on entities and regex
newtopic.create_binary_labels() # creates the actual labels that will be used in training


# In[11]:


# Examine a sample of positively labeled cases 
for line in newtopic.data[newtopic.data.positive_label==True].sample(5).textcol:
    print(line+ "\n")


# In[12]:


for line in newtopic.data[newtopic.data.negative_label==True].sample(5).textcol:
    print(line + "\n")


# ### Create the training data

# In[13]:


newtopic.create_training_data() # removes rows that fall into both positive and negative categories or neither


# ### Create the model

# In[14]:


newtopic.create_model() # actually runs a TF-IDF, then the SVM model itself on the training data


# ### Show informative model features

# In[15]:


newtopic.show_most_informative_features()


# ### Cross-validate the model and return CV accuracy

# In[16]:


newtopic.cross_val()


# In[17]:


newtopic.has_tfidf


# In[18]:


newtopic.k_word_cross_val()


# In[19]:


pos_regex_k = pos_regex.split("|")


# In[ ]:


for k in pos_regex_k:
    pos_regex_minus_k = [x for x in pos_regex_k if x != k]
    pos_regex_minus_k = "|".join(pos_regex_minus_k)
    newtopic.define_positive(pos_regex = pos_regex_minus_k)
    newtopic.create_binary_labels()
    newtopic.create_training_data()
    newtopic.create_model()


# ### Show the 'delta', how many new posts are found by the model

# In[ ]:


# What proportion of the original data were labeled as positive?
newtopic.label_based_proportion


# In[ ]:


newtopic.model_based_increase


# In[ ]:


# Show the total
newtopic.label_based_proportion + newtopic.model_based_increase


# In[ ]:


len(newtopic.data)


# In[ ]:


# Number of newly-labeled political posts, not from source set
len(newtopic.data[(newtopic.data.predicted_labels==True) & (newtopic.data.positive_label==False)])


# In[ ]:


# Print out a sample of the text that was 'found' by the model to be political, but didn't contain our seed set
for line in newtopic.data[(newtopic.data.predicted_labels==True) & (newtopic.data.positive_label==False)].sample(10).textcol:
    print(line + "\n")

