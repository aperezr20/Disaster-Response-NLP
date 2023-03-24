import sys
import pandas as pd

import sqlite3
import pickle


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import os

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    
    # establish a connection to the SQLite database
    conn = sqlite3.connect(database_filepath)
    

    #df = pd.read_sql_query("SELECT * FROM table", conn)

    
    # read the table into a pandas DataFrame
    df = pd.read_sql_query("SELECT * FROM table_disaster", conn)
    
    #df = pd.read_sql("SELECT * FROM Message", engine)
    df = df.replace(2,1)
    
    categories = list(df.columns[5:])
    X = df[['message']] 
    y = df[categories]
    
    return X, y, categories

def tokenize(text):
    # Tokenizer
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
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators':[10,50,100],
        'clf__estimator__criterion':['gini', 'entropy'],
        'clf__estimator__max_depth':[None, 10, 20]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
    
    


def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test.values.flatten())
    
    report = classification_report(Y_test.astype(int), y_pred.astype(int), target_names=category_names)
    
    print(report)
    
    return report
    


def save_model(model, model_filepath):
    pickle.load(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train.values.flatten(), Y_train)
        
        print('Best Parameters')
        print(model.best_params_)
        
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