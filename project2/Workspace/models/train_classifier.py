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
    engine = create_engine('sqlite:///' + database_filepath)
    conn=engine.connect()
    #query = "SELECT * FROM merged_df"
   # df = pd.read_sql(query, engine)
    df = pd.read_sql_table('merged_df', conn)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    return X,Y
   


def tokenize(text):
    """ tokenizes the input text,
      removes punctuation, 
      converts words to lowercase,
        removes stop words, and then performs lemmatization on the remaining tokens
    """
    punctuation = '''!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'''
    tokens = []
    words = text.split()
    for word in words:
        # Remove punctuation marks from the word
        word = ''.join([char for char in word if char not in punctuation])
        # Convert the word to lowercase and add it to the list of tokens
        tokens.append(word.lower())

     #remove stop words
     
    word = [w for w in word if w not in stopwords.words("english")]
        
    # apply Lemmatization
    tokens = [WordNetLemmatizer().lemmatize(w) for w in word]    
    
    return tokens


def build_model():
   
    #vectorizer = CountVectorizer()
    #tfidf_transformer = TfidfTransformer()
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
    Y_pred = model.predict(X_test)

# Report f1 score, precision, and recall for each output category
    for i, col in enumerate(Y_test.columns):
        print(classification_report(Y_test[col], Y_pred[:, i]))

def save_model(model, model_filepath):
    with open('Tmodel.pkl', 'wb') as file:
        pickle.dump(model, file)
    
        
def main():
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
        
