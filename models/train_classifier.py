import sys
import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, recall_score, precision_score
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import joblib

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
    """
    Custom tokenizer for message text

    Messages are normalized to lower case, stopwords are removed and the remaining words are lemmatized using
    WordNet lemmatizer.

    :param text: text to tokenize
    :return: list of lowercase, lemmatized tokens
    """
    tokens = [tok.lower() for tok in word_tokenize(text, language='english') if tok not in stopwords.words("english")]
    lemmas = [WordNetLemmatizer().lemmatize(tok) for tok in tokens]
    return lemmas


def build_model():
    """
    Message Classification pipeline builder

    The Pipeline is made up of
     1. TF-IDF Vectorizer using the custom tokenize function from this module
     2. Multi-output classifier using a Randon Forest classifier

    The pipeline is set up for grid search of optimized parameters. The optimized parameters are
     * vectorizer ngram range, 1-grams or 1- and 2-grams
     * maximum number of features extracted by the vectorizer, 50-200
     * number of estimators in the random forest classifier, 50-200 estimators

    :return: message classification pipeline
    """
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(tokenizer=tokenize)),
        ('classifier', MultiOutputClassifier(RandomForestClassifier(), n_jobs=-1)),
    ])
    model = GridSearchCV(pipeline, {
        'vectorizer__ngram_range': [(1, 1), (1, 2)],
        'vectorizer__max_features': [50, 100, 200],
        'classifier__estimator__n_estimators': [50, 100, 200]
    }, n_jobs=None)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Model evaluation and output to console.

    For context, the optimized model parameters are printed frist.
    Weighted average f1 score, precision and prediction are tabtabulated per category

    :param model: model to evaluate
    :param X_test: test set messages
    :param Y_test: test set true categories
    :param category_names: list of category names
    """
    y_pred = model.predict(X_test)
    print("Model optimal paramaters")
    for par in model.best_params_:
        print(par, " : ", model.best_params_[par])
    print("\nModel evaluation")
    print("category                  | f1 score   | precision  | recall    ")
    print("================================================================")
    for i, category in enumerate(category_names):
        print("{0:25s} | {1:10.6f} | {2:10.6f} | {3:10.6f}".format(
            category, f1_score(Y_test[category].values, y_pred[:,i], average='weighted'),
            precision_score(Y_test[category].values, y_pred[:,i], average='weighted'),
            recall_score(Y_test[category].values, y_pred[:,i], average='weighted')
        ))
    print("================================================================")


def save_model(model, model_filepath):
    """
    Efficiently saves model in pickle format at the specified file path

    :param model: model to save
    :param model_filepath: path to pickle file to create
    """
    joblib.dump(model, model_filepath)


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
