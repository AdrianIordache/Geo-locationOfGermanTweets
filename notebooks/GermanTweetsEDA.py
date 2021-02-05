#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display

import warnings
warnings.filterwarnings("ignore")

plt.style.use('seaborn-poster')


# In[2]:


COLUMNS = ["id", "latitude", "longitude", "text"]

PATH_TO_TRAIN = "/home/adrian/Desktop/Python/Personal/Kaggle/Geo-locationOfGermanTweets/data/training.txt"
PATH_TO_VALID = "/home/adrian/Desktop/Python/Personal/Kaggle/Geo-locationOfGermanTweets/data/validation.txt"


# In[3]:


train_df = pd.read_csv(PATH_TO_TRAIN, sep = ',', header = None)
train_df.columns = COLUMNS

valid_df = pd.read_csv(PATH_TO_VALID, sep = ',', header = None)
valid_df.columns = COLUMNS


# In[4]:


display(train_df)
display(valid_df)


# In[5]:


data = pd.concat([train_df, valid_df], axis = 0)
data = data.sample(frac = 1).reset_index(drop=True)


# # Statistics and Distributins for latitude and longitude

# In[6]:


rd = lambda x: np.round(x, 3)

latitude = data.latitude.values
longitude = data.longitude.values

print("Latitude  -> Min {}, Max {}, Mean {}, Std {}".format(rd(np.min(latitude)), rd(np.max(latitude)), rd(np.mean(latitude)), rd(np.std(latitude))))
print("Longitude -> Min {}, Max {}, Mean {}, Std {}".format(rd(np.min(longitude)), rd(np.max(longitude)), rd(np.mean(longitude)), rd(np.std(longitude))))


# In[7]:


plt.figure(figsize = (14, 10))
plt.hist(latitude, bins = 50)
plt.title("Distribution of latitude in data")
plt.xlabel("Latitude")
plt.ylabel("Count")
plt.show()


# In[8]:


plt.figure(figsize = (14, 10))
plt.hist(longitude, bins = 50)
plt.title("Distribution of longitude in data")
plt.xlabel("Longitude")
plt.ylabel("Count")
plt.show()


# In[9]:


plt.figure(figsize = (20, 30))
for (i, bins) in enumerate([3, 5, 10, 20, 35, 50]):
    ax = plt.subplot(3, 2, i + 1)
    ax.hist(latitude, bins = bins)
    ax.set_title("Distribution of latitude in data with {} bins".format(bins))
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Count")
    
plt.show()


# In[10]:


plt.figure(figsize = (20, 30))
for (i, bins) in enumerate([3, 5, 10, 20, 35, 50]):
    ax = plt.subplot(3, 2, i + 1)
    ax.hist(longitude, bins = bins)
    ax.set_title("Distribution of longitude in data with {} bins".format(bins))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Count")
    
plt.show()


# # Tweet Lengths Distribution

# In[11]:


data["tweet_length"] = data["text"].apply(lambda x: len(str(x)))

display(data)


# In[12]:


tweet_length = data.tweet_length.values

print("Tweet Lengths  -> Min {}, Max {}, Mean {}, Std {}".format(np.min(tweet_length), np.max(tweet_length), rd(np.mean(tweet_length)), rd(np.std(tweet_length))))


# In[13]:


plt.figure(figsize = (20, 30))
for (i, bins) in enumerate([5, 10, 20, 35, 50, 100]):
    ax = plt.subplot(3, 2, i + 1)
    ax.hist(tweet_length, bins = bins)
    ax.set_title("Distribution of tweets lengths in data with {} bins".format(bins))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Count")
    
plt.show()


# In[14]:


values, counts = np.unique(tweet_length, return_counts = True)

print("Number of unique tweet lenghts: {}".format(len(values)))


# # From continuous to discrete 

# In[15]:


latitude_bins      = np.digitize(latitude,     bins = [48, 49, 50, 51, 52, 53])
longitude_bins     = np.digitize(longitude,    bins = [7, 8, 9, 10, 11, 12])
tweet_length_bins  = np.digitize(tweet_length, bins = [100, 200, 300, 400, 500, 600, 700])


# In[16]:


assert len(latitude_bins) == len(longitude_bins) == len(tweet_length_bins) == data.shape[0], "Something went wrong"


# In[17]:


data["latitude_bins"]     = latitude_bins
data["longitude_bins"]    = longitude_bins 
data["tweet_length_bins"] = tweet_length_bins

display(data)


# In[18]:


group_by_latitude = data.groupby("latitude_bins")["longitude"].describe()
display(group_by_latitude)

group_by_latitude = data.groupby("latitude_bins")["tweet_length"].describe()
display(group_by_latitude)


# In[19]:


group_by_longitude = data.groupby("longitude_bins")["latitude"].describe()
display(group_by_longitude)

group_by_longitude = data.groupby("longitude_bins")["tweet_length"].describe()
display(group_by_longitude)


# In[20]:


group_by_tweet_count = data.groupby("tweet_length_bins")["latitude"].describe()
display(group_by_tweet_count)

group_by_tweet_count = data.groupby("tweet_length_bins")["longitude"].describe()
display(group_by_tweet_count)


# In[21]:


PATH_TO_PREPROCCESED_TRAIN = "/home/adrian/Desktop/Python/Personal/Kaggle/Geo-locationOfGermanTweets/data/preprocessed/preprocessed_train.csv"
PATH_TO_PREPROCCESED_VALID = "/home/adrian/Desktop/Python/Personal/Kaggle/Geo-locationOfGermanTweets/data/preprocessed/preprocessed_valid.csv"


# In[22]:


preproccesed_train = pd.read_csv(PATH_TO_PREPROCCESED_TRAIN)
preproccesed_valid = pd.read_csv(PATH_TO_PREPROCCESED_VALID)

preproccesed_data = pd.concat([preproccesed_train, preproccesed_valid], axis = 0)
preproccesed_data = preproccesed_data.sample(frac = 1).reset_index(drop=True)


# In[23]:


display(preproccesed_data)


# In[24]:


preproccesed_data["tweet_length"]       = preproccesed_data["text"].apply(lambda x: len(str(x)))
preproccesed_data["final_tweet_length"] = preproccesed_data["final_text"].apply(lambda x: len(str(x)))
preproccesed_data["length_diff"]        = preproccesed_data["tweet_length"] - preproccesed_data["final_tweet_length"] 

display(preproccesed_data)


# In[25]:


length_diff = preproccesed_data.length_diff.values

print("Tweet Lengths Diff -> Min {}, Max {}, Mean {}, Std {}".format(np.min(length_diff), np.max(length_diff), rd(np.mean(length_diff)), rd(np.std(length_diff))))


# In[3]:


PATH_TO_LEVEL_ONE_FEATURES = "/home/adrian/Desktop/Python/Personal/Kaggle/Geo-locationOfGermanTweets/data/level_two/level_two_features.csv"
level_one_features = pd.read_csv(PATH_TO_LEVEL_ONE_FEATURES)

display(level_one_features)


# In[4]:


latitude  = level_one_features.latitude.values
longitude = level_one_features.longitude.values

latitude_bins      = np.digitize(latitude,     bins = [48, 49, 50, 51, 52, 53])
longitude_bins     = np.digitize(longitude,    bins = [7, 8, 9, 10, 11, 12])

level_one_features["latitude_skf"]  = latitude_bins
level_one_features["longitude_skf"] = longitude_bins

display(level_one_features)

PATH_TO_LEVEL_ONE_FEATURES_SKF = "/home/adrian/Desktop/Python/Personal/Kaggle/Geo-locationOfGermanTweets/data/level_two/level_two_features_skf.csv"
level_one_features.to_csv(PATH_TO_LEVEL_ONE_FEATURES_SKF, index = False)

