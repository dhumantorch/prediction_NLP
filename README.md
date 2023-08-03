# Natural Language Processing

This project is an example of NLP on Tripadvisor hotel reviews with ratings from 1 to 5.  The data is pretty clean (there's more data cleaning in my descriptive MLR project so far), but I prep the data corpus 
using typical methods such as getting rid of characters, stopwords, and stemming each word.  I leave most negative words in the data because "I will definitely come back" has a far different connotation than 
"I will definitely not come back," in terms of sentiment analysis.

I wrote functions to compare multiple word counts for the bag of words sparse matrix, using **Random Forest** with varying numbers of trees and a couple different criteria, and also compared **XGBoost**, both in terms 
of model accuracy and runtime, so there's some model selection there, as well as k-fold cross-validation to make sure we're not being too optimistic.

The best model I came up with was XGBoost using a 750-word bag of words, which got about a 59% accuracy rate.  Pure guessing would have gotten about a 20% accuracy, and knowing that 5-star reviews were 44% of 
the total reviews would have gotten a 44% accuracy by just guessing 5s, so 59% is decent for this model, especially given the ambiguity of reviews vs ratings.  I would definitely like to improve the percentage.

# Setup
You can read the ipynb file on here, or download it to test the code.  For any of the imports you don't have, you can run, for example, !pip install xgboost within the notebook.  See requirements below.

# Requirements:
python
jupyter notebook

#Standard python libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Libraries for NLP
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

#Libraries for Random Forest prediction model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import time

#Libraries for XGBoost
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

# *Dataset taken from Kaggle: https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews
# Requested Citation:
Citation
Alam, M. H., Ryu, W.-J., Lee, S., 2016. Joint multi-grain topic sentiment: modeling semantic aspects for online reviews. Information Sciences 339, 206–223. DOI

License
CC BY NC 4.0

Splash banner
Photo by Rhema Kallianpur on Unsplash.

Splash icon
Logo by Tripadvisor.
