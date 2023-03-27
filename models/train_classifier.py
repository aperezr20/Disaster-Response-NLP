import os
import sys
import pickle
import sqlite3
import multiprocessing

import pandas as pd
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    Load and preprocess data from a SQLite database.

    Args:
    database_filepath (str): Filepath to SQLite database

    Returns:
    X (DataFrame): DataFrame containing the feature data
    y (DataFrame): DataFrame containing the target data
    categories (list): List of category names
    """
    
    # establish a connection to the SQLite database
    conn = sqlite3.connect(database_filepath)
    df = pd.read_sql_query("SELECT * FROM table_disaster", conn)

    # replace 2 with 1
    df = df.replace(2, 1)

    categories = list(df.columns[5:])
    X = df['message']
    y = df[categories]

    return X, y, categories


def tokenize(text):
    """
    Tokenize and preprocess text data.

    Args:
    text (str): Text to be tokenized and preprocessed

    Returns:
    clean_tokens (list): List of cleaned tokens
    """

    # tokenizer
    tokens = word_tokenize(text.lower())
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok)
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    Build a pipeline and perform a grid search over different models and
    hyperparameters.

    Returns:
    cv (GridSearchCV): A grid search object with a pipeline
    """
    
    # classifiers for grid search
    clf_models = {
        'Random Forest': [RandomForestClassifier()],
        'Perceptron': [MLPClassifier()],
    }

    # classifiers' parameters for grid search
    clf_params = [
        {
            'clf__estimator': clf_models['Random Forest'],
            'clf__estimator__n_estimators': [50, 100]
        },
        {
            'clf__estimator': clf_models['Perceptron'],
            'clf__estimator__activation': ['tanh', 'relu'],
            'clf__estimator__hidden_layer_sizes':[10],
            'clf__estimator__max_iter': [300]
        }
    ]
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])

    # F1 scorer initialization for grid search
    scorer = make_scorer(f1_score, zero_division=0, average='macro')

    cv = GridSearchCV(
        pipeline,
        cv=2,
        param_grid=clf_params,
        verbose=5,
        n_jobs=-1,
        scoring=scorer
    )

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate the performance of the text classification model and print a classification report.
    
    Args:
        model (object): The trained model.
        X_test (DataFrame): The test data containing message data.
        Y_test (DataFrame): The test data containing target data.
        category_names (list): The list of category names for target data.

    Returns:
        str: Returns the classification report as a string.
    
    """

    y_pred = model.predict(X_test)
    
    report = classification_report(Y_test.astype(int), y_pred.astype(int), target_names=category_names)
    
    print(report)
    
    return report
    

def save_model(model, model_filepath):
    """Save the trained model as a pickle file.
    
    Args:
        model (object): The trained model object to be saved.
        model_filepath (str): The file path where the model object should be saved.

    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """Main function to execute the pipeline."""

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
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