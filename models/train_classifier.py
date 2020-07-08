import sys
import pandas as pd
import sqlite3

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

def load_data(database_filepath):
    """
    Read previously ingested training data from provided sqlite3 database file.

    :param database_filepath: path to the database file to read from
    :return: factors, responses, response category names
    """
    with sqlite3.connect(database_filepath) as conn:
        raw_data_df = pd.read_sql_query('SELECT * FROM disaster_messages;', conn, index_col='id')
    catnames = list(raw_data_df.columns)
    for remcat in ['message', 'original', 'genre']:
        catnames.remove(remcat)
    Y = raw_data_df[catnames]
    X = raw_data_df['message']
    return X, Y, catnames


def tokenize(text):
    tokens = [tok.lower() for tok in word_tokenize(text, language='english') if tok not in stopwords.words("english")]
    lemmas = [WordNetLemmatizer().lemmatize(tok) for tok in tokens]
    return lemmas


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        print(X)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()