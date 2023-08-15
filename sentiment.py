#In order to run this code you first Need to go to the bottom and enter a query to determine which tweets you want to scrape.
#There is an example query provided in the comments above the command available to copy and paste for a test

import snscrape.modules.twitter as sntwitter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

plt.style.use('ggplot')

#if Vader lexicon is missing download it here
#potential solution to NLTK issue
nltk.download('vader_lexicon')

#Scrape all tweets from a user depending on the query given
#Takes quite a while beware
def scrapeTweets(query, tweet_num, robAnalysis, vaderAnalysis, simplfySentiment):
    tweets = []
    #iterates through all tweets that match the query selected and appends them to an array
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets) == tweet_num:
            break
        tweets.append([tweet.id, tweet.date, tweet.likeCount, tweet.retweetCount, tweet.replyCount, 
                        tweet.quoteCount, tweet.content, tweet.media])
    df = pd.DataFrame(tweets, columns = ['ID', 'Date', 'Likes', 'Retweets', 'Replies', 
                        'Quotes', 'Content', 'Media'])
		#if any type of analysis is set to true in an if statement will analyze the dataset in addition 
		#to scraping the posts
    if robAnalysis == True and vaderAnalysis == True:
        df = addRobertaSentiment(df, simplfySentiment)
        df = addVaderSentiment(df, simplfySentiment)
    elif (robAnalysis == True):
        df = addRobertaSentiment(df, simplfySentiment)
    elif (vaderAnalysis == True):
        df = addVaderSentiment(df, simplfySentiment)
    df.to_csv('Sentiment_File.csv')
    return df

#Analyze the tweets using Vader lexicon
def addVaderSentiment(df, simple):
    results = {}
    sia = SentimentIntensityAnalyzer()
		#iterates post by post and pass the post to the given model and return the three 
		#sentiment values used to judge the model on
    for i, row in tqdm(df.iterrows(), total = len(df)):
        try:
            text = row['Content']
            my_id = row['ID']
            vader_results = sia.polarity_scores(text)
            vader_results_rename = {}
            for key,value in vader_results.items():
                vader_results_rename[f"Vader {key}"] = value
            if simple == True:
                vaderComponents = {**vader_results_rename}
                results[my_id] = simpleSentiment(vaderComponents.get('Vader neg'), vaderComponents.get('Vader neu'), vaderComponents.get('Vader pos'),vaderComponents.get('Vader compound'))
            else:
                results[my_id] = {**vader_results_rename}
        except RuntimeError:
						#if there is an error thrown for a specific tweet ID it will print the Tweet ID of the broken post
            print(f'Broke for id: {my_id}')
    if simple == True:
        sentiment_df = pd.DataFrame(results, index = ["Vader Sentiment"]).T
    else:
        sentiment_df = pd.DataFrame(results).T
    sentiment_df = sentiment_df.reset_index().rename( columns = {'index':'ID'})
    sentiment_df = sentiment_df.merge(df, how = 'right')
    return sentiment_df


#Takes quite a while beware
#Must have tweet ID in passed df for merging
def addRobertaSentiment(df, simple):
    results = {}
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
		#iterates post by post and pass the post to the given model and returns the three 
		#sentiment values used to judge the model on
    for i, row in tqdm(df.iterrows(), total = len(df)):
        try:
            text = row['Content']
            my_id = row['ID']
            tokens = tokenizer(text, return_tensors = 'pt')
            rob_results = polarityScoresRoberta(tokens, model)
            if simple == True:
                robComponents = {**rob_results}
                results[my_id] = simpleSentiment(robComponents.get('Roberta neg'), robComponents.get('Roberta neu'), robComponents.get('Roberta pos'), 0.00)
            else:
                results[my_id] = {**rob_results}
        except RuntimeError:
						#if there is an error thrown for a specific tweet ID it will print the Tweet ID of the broken post
            print(f'Broke for id: {my_id}')
    if simple == True:
        sentiment_df = pd.DataFrame(results, index = ["Roberta Sentiment"]).T
    else:
        sentiment_df = pd.DataFrame(results).T
    sentiment_df = sentiment_df.reset_index().rename( columns = {'index':'ID'})
    sentiment_df = sentiment_df.merge(df, how = 'right')
    return sentiment_df

def polarityScoresRoberta(tokens, model):
		#runs the more complicated modeling for the Roberta model and returns a dict
		#of the values returned by the model to be used for analysis
    output = model(**tokens)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {'Roberta neg':scores[0], 
                   'Roberta neu':scores[1],
                   'Roberta pos':scores[2]
                  }
    return scores_dict

def simpleSentiment(neg, neu, pos, comb):
    if(comb != 0):
				#Returns Positive if combined sentiment from the Vader model is > 0.33
				#Returns Neutral if combined sentiment from the Vader model is between -0.33 and 0.33
				#Returns Negative if the above conditions are not fulfilled
        if comb > 0.33:
            return 'Positive'
        elif -0.33 < comb < 0.33:
            return 'Neutral'
        else:
            return 'Negative'
		#if the Positive value from the Roberta model is larger than returns a Positive sentiment
    elif (pos > neu) and (pos > neg):
        return 'Positive'
		#if the negative value from the Roberta model is the largest it returns a Negative sentiment
    elif neg > neu and neg > pos:
        return 'Negative'
		#Returns a Neutral sentiment if the above conditions are not fulfilled
    else:
        return 'Neutral'
    
#A query that the scraper uses to determine the subject that needs to be gathered.
#e.g. query = "(from:elonmusk) until:2023-1-1 since:2013-12-06"
#The above query example scrapes all of Elon Musks' tweets from December 6th, 2013 until
# the end of 2022
#A query can be formed by using twitters advanced search to find a topic and copy the resulting query 
#it gives from the search bar at the top
query = ""

#The value in tweet num determines how many tweets will be returned by scraping
#e.g. tweet_num = 10 will create a CSV with 10 tweets scraped from the given query
tweet_num = 10

#These boolean analysis variables determine if sentiment analysis will be done on the scraped tweets
#or just returning a CSV with just the tweets sans sentiment
rob_analysis = True
vader_analysis = True

#Deterines whether sentiment will be just simplified to just negative, neutral or positive
#instead of more precise numeric values
simple_sentiment = True
try:
    df = scrapeTweets(query, tweet_num, rob_analysis, vader_analysis, simple_sentiment)
except Exception as e:
    print("Check if the query is filled in correctly")
    print(e)