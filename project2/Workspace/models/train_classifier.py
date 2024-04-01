import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
        """    
          the functions load file from the database
            the parameter database_filepath gives location and name of the loaded database 
            returning X and Y variables
        """
    engine = create_engine('sqlite:///' + database_filepath)
    conn=engine.connect()
    #query = "SELECT * FROM merged_df"
   # df = pd.read_sql(query, engine)
    df = pd.read_sql_table('merged_df', conn)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    return X,Y

def tokenize(text):
    
    """
    tokenizes the input text,
    removes punctuation, 
    converts words to lowercase,
    removes stop words,
    and then performs lemmatization on the remaining tokens
    returning lemmatized tokens
    """
    punctuation = '''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'''
    tokens = []
    words = text.split()
    for word in words:
        # Remove punctuation marks from the word
        word = ''.join([char for char in word if char not in punctuation])
        # Convert the word to lowercase and add it to the list of tokens
        tokens.append(word.lower())

    # remove stop words
    
    tokens = [w for w in tokens if w not in stopwords.words("english")]
        
    # apply Lemmatization
    tokens = [WordNetLemmatizer().lemmatize(w) for w in tokens]    
    
    return tokens

def build_model():
    """    
    the function builds a pipeline, 
    retuning a Cv which equates to a Gridsearch
    """
    # creating a Machine Learning pipeline        
    pipeline = Pipeline([
    ('vect', CountVectorizer()),  # Text vectorization
    ('tfidf', TfidfTransformer()),  # TF-IDF transformation
    ('clf', MultiOutputClassifier(RandomForestClassifier()))  # MultiOutputClassifier with RandomForestClassifier
])
    
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    #pipeline.fit(X_train, Y_train)
    #Y_pred = pipeline.predict(X_test)
    
    params = {
      
       'clf__estimator__n_estimators': [50, 100],

    }
               
    cv = GridSearchCV(pipeline, param_grid=params,cv=3,n_jobs=-1,scoring="accuracy")
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
        The fucntion evaluates the performance of the trained model.
        Parameters: 
            model : The trained model under evaluation
            X_test : Features for testing
            Y_test :  test data
            category_names : List of category names
            
    """
    Y_pred = model.predict(X_test)

# Report f1 score, precision, and recall for each output category
    for i, col in enumerate(Y_test.columns):
        print(classification_report(Y_test[col], Y_pred[:, i]))

def save_model(model, model_filepath):
        """
        Save the trained model to a pickle file.
        Parameters:
        model (): The trained model to be saved.
        model_filepath (str): The filepath where the model will be saved.

        """
        with open(model_filepath, 'wb') as file:
            pickle.dump(model, file)
    
        
def main():
        """

            Load data from a database file, train a classifier, evaluate its performance,
             and save the trained model to a file.

            This function expects two command-line arguments:
            - database_filepath: The path to the SQLite database file containing the data.
            - model_filepath: The path to save the trained model as a pickle file.

        """

if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        category_names = Y.columns.tolist()
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
        
