
from __future__ import division
import sys, pandas, pickle, numpy, scipy, itertools, os, django, datetime, random
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import numpy as np
from time import time
import re
from itertools import compress

# Things I want this to do:
# - tell me what the accuracy/precision/recall is of the SVM
# - tell me how many more documents we have by using SVM (percent increase/decrease over just using regex alone)
# Define a class
class Svmtopic:
    def __init__(self, data, text_column, entities_column, positive_regex, negative_regex, positive_entities, strip_src_text):
        self.data = data
        self.text_column = text_column
        self.entities_column = entities_column
        self.positive_entities = positive_entities
        self.positive_regex = positive_regex
        self.negative_regex = negative_regex
        self.strip_src_text = strip_src_text

    def define_positive(self):
      try:
        self.data["textcol"] = self.data.textcol.str.lower()
        self.data["entity_based_positive_rows"] = self.data[self.entities_column].isin(self.positive_entities)
        self.data["regex_based_positive_rows"] = self.data[self.text_column].str.contains(self.positive_regex, na=False)
        self.data["regex_based_negative_rows"] = self.data[self.text_column].str.contains(self.negative_regex, na=False)
      except Exception as e:
        print(e.__doc__)
        print(e.message)

    def create_binary_labels(self):
        # this depends on the define_positive, so run if missing:
      if not "entity_based_positive_rows" in self.data.columns:
        self.define_positive()
      print("Creating labels")
      self.data["positive_label"] = (self.data.entity_based_positive_rows) & (self.data.regex_based_positive_rows) & (self.data.regex_based_negative_rows==False)
      print('positive labels', self.data["positive_label"].unique())
      self.data["negative_label"] = self.data.regex_based_negative_rows==True
      print('negative labels', self.data["negative_label"].unique())
    
    def create_training_data(self):
      if not 'positive_label' in self.data.columns:
        self.create_binary_labels()
      # Remove excessively short strings
      self.df_train = self.data[[len(s) > 10 for s in self.data[self.text_column].values] ]
      # Remove items that are not explicitly hard or soft from the training data:
      self.df_train = self.df_train[(self.df_train.positive_label) | (self.df_train.negative_label)]
      # reset the index
      self.df_train = self.df_train.reset_index(drop=True)
      if self.strip_src_text:
        # strip out the regex text itself - the goal here is to expand the set
        text_tokenized = [str(x.encode('utf-8')).split() for x in self.df_train[self.text_column]]
        pos_pattern = re.compile(self.positive_regex)
        # keep those that return None, because that means the regex is missing from the word
        words_to_keep = [[pos_pattern.search(x)==None for x in sublist] for sublist in text_tokenized]
        # remove those positively tagged words, and make it the clean text
        clean_text =[]
        for posts,to_keep in zip(text_tokenized, words_to_keep):
            tmp_list = list(compress(posts, to_keep))
            tmp_joined = " ".join(tmp_list)
            clean_text.append(tmp_joined)
        self.df_train["clean_text"] = clean_text
      else:
        self.df_train["clean_text"] = self.df_train["textcol"]
         
    def create_model(self):
      if not hasattr(self, 'df_train'):
        self.create_training_data()
      print "Extracting features from the training dataset using a sparse vectorizer"
      self.vectorizer = TfidfVectorizer(
        sublinear_tf=False,
        max_df=0.5,
        min_df=2,
        ngram_range = (1,3),
        use_idf=False,
        stop_words='english')
      self.X_traintfidf = self.vectorizer.fit_transform(self.df_train['clean_text'].values)
      self.X_teststfidf = self.vectorizer.transform(self.data[self.text_column].values)
      # Now fit the model:
      #from sklearn.calibration import CalibratedClassifierCV
      self.labels = self.df_train['positive_label'].values
      print "Running SVM...."
      self.clf = LinearSVC(loss='hinge', penalty='l2', C=1, random_state = 7)
      #clf = CalibratedClassifierCV(clf_unc) 
      self.clf.fit(self.X_traintfidf, self.labels)
      self.data['predicted_labels'] = self.clf.predict(self.X_teststfidf)
      # print out the original label proportion
      self.label_based_proportion = sum(self.data.positive_label)/len(self.data)
      # print out the proportion of new labels based on the svm
      self.model_based_increase = (sum(self.data.predicted_labels)-sum(self.data.positive_label))/len(self.data)
      self.train_preds = self.clf.predict(self.X_traintfidf)
      print(metrics.classification_report(self.labels, self.train_preds))
    def cross_validate(self):
      if not hasattr(self, 'clf'):
        self.create_model()
      self.scores = cross_val_score(
      self.clf, self.X_traintfidf, self.labels, cv=10, scoring= 'accuracy')
      print "CV Accuracy: %0.3f (+/- %0.3f)" % (self.scores.mean(), self.scores.std() / 2)

    def show_most_informative_features(self, n=20):
      if not hasattr(self, 'vectorizer'):
        self.create_model()
      c_f = sorted(zip(self.clf.coef_[0], self.vectorizer.get_feature_names()))
      top = zip(c_f[:n], c_f[:-(n+1):-1])
      for (c1,f1),(c2,f2) in top:
        print "\t%.4f\t%-15s\t\t%.4f\t%-15s" % (c1,f1,c2,f2)
