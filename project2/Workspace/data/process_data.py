import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import sqlite3
import sqlalchemy

def load_data(messages_filepath, categories_filepath):
    """ 
                Loads the two files , merges the two files  and produce one merged file
                param1 messages_filepath - file taht contains messages
                param2 categories_filepath - file contains categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    merged_df = pd.merge(messages, categories, on='id', how='inner')
    categories_split = merged_df['categories'].str.split(';', expand=True)
    row = categories_split.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories_split.columns = category_colnames
    for column in categories_split:
    # Check if the column starts with 'category'
   ## if column.startswith('category_split'):
        # Extract the last character of each string and convert it to numeric
        categories_split[column] = categories_split[column].astype(str).str[-1].astype(int)  
    merged_df = pd.concat([merged_df.drop('categories', axis=1), categories_split], axis=1)
    #merged_df.drop_duplicates(inplace=True)
    merged_df=merged_df[merged_df["related"]!=2]
    return merged_df

def clean_data(merged_df):
    merged_df.drop_duplicates(inplace=True)
    return merged_df
    
    
def save_data(merged_df, database_filepath):
    #Import the create_engine function from the SQLAlchemy library
    from sqlalchemy import create_engine

    # Create an SQLAlchemy engine to connect to the SQLite database
    engine = create_engine(f'sqlite:///{database_filepath}')

    # Write the DataFrame to a SQL database table named 'merged_df'
    merged_df.to_sql('merged_df', engine, index=False, if_exists="replace")
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()