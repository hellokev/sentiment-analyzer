import re # Regular Expression
import argparse # Command Line Argument
from tqdm import tqdm # Progress Bar for Python and CLI
import pandas as pd
import nltk
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def addSentimentAnalysis(filepath, column_name):
    """Open a CSV file and add sentiment analysis using 
                Vader and roBERTa-base model
    Args:
        filepath (string): A path to the CSV file
    """

    # This is a roBERTa-base model trained on ~58M tweets and finetuned for 
    # sentiment analysis with the TweetEval benchmark. 
    # This model is suitable for English (for a similar multilingual model, see XLM-T).
    # Link: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
    MODEL = 'cardiffnlp/twitter-roberta-base-sentiment'
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    sia = SentimentIntensityAnalyzer()

    try:
        df = pd.read_csv(filepath)

        # Check user input:
        if column_name not in df.columns.values:
            raise Exception(f'Error: The column "{column_name}" does not exist')

        # Add columns if not exists
        if 'RobertaSentiment' not in df.columns.values:
            df.insert(2, 'RobertaSentiment', 'N/A')
        
        if 'VaderSentiment' not in df.columns.values:
            df.insert(3, 'VaderSentiment', 'N/A')

        for index, row in tqdm(df.iterrows(), total = len(df)):
            #Roberta Sentiment
            tokens = tokenizer(row[column_name], return_tensors = 'pt')
            model_scores = polarityScoresRoberta(tokens, model)
            df.at[index, 'RobertaSentiment'] = simpleRobertaSentiment(**model_scores)

            #Vader Sentiment
            model_scores = sia.polarity_scores(row[column_name])
            df.at[index, 'VaderSentiment'] = simpleVaderSentiment(model_scores.get('compound'))

        df.to_csv(filepath, index = False)
    except Exception as e:
        print(e)

def polarityScoresRoberta(tokens, model):
    output = model(**tokens)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {'neg':scores[0], 
                   'neu':scores[1],
                   'pos':scores[2]
                  }
    return scores_dict

def simpleRobertaSentiment(neg, neu, pos):
    if pos > neu and pos > neg:
        return 'Positive'
    elif neg > neu and neg > pos:
        return 'Negative'
    else:
        return 'Neutral'

def simpleVaderSentiment(compound):
    if compound > 0.33:
        return 'Positive'
    elif compound < -0.33:
        return 'Negative'
    else:
        return 'Neutral'
    
if __name__ == "__main__":
    # Command Line Argument
    parser = argparse.ArgumentParser(description="Sentiment Analysis")
    parser.add_argument('path', metavar='path', type=str, help="Twitter search query")
    parser.add_argument('column', metavar='column', type=str, help="Column")
    args = parser.parse_args()

    # Download or update the required package for tokenization
    nltk.download('vader_lexicon')

    # Run
    addSentimentAnalysis(args.path, args.column)