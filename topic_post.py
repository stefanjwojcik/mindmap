
import os
import pandas as pd
import svm_topic

sample_posts = pd.read_csv("facebook_posts_sample.csv", encoding = 'utf-8', sep = '\t')

# Any text containing one of these fragments is considered 'political', and labeled positively if it came from hard news source 
pos_regex = 'politi|usnews|world|national|state|elect|vote|govern|campaign| war |polic|econ|unemploy|racis|energy|abortion|educa|healthcare|immigration|weather'
# Any text containing one of these fragments is considered NOT political
neg_regex = 'ncaa|sports|entertainment|arts|fashion|style|lifestyle|leisure|celeb|movie|music|gossip|food|travel|horoscope|gadget'

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

newtopic = svm_topic.Svmtopic(sample_posts, text_column = 'textcol',
                    entities_column = 'publisher_name',
                    positive_regex = pos_regex, 
                    negative_regex = neg_regex, 
                    positive_entities = pos_entities, 
                    strip_src_text = False)


newtopic.k_word_cross_val()




