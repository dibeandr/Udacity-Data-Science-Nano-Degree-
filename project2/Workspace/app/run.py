import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar, Histogram
from flask import Flask, render_template, request
from plotly.graph_objs import Bar as PlotlyBar  # Rename Plotly's Bar class to avoid conflicts
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Load data
engine = create_engine('sqlite:///data/PipelineDataClean.db')
df = pd.read_sql_table('merged_df', engine)

# Load model
model = joblib.load("models/Tmodel.pkl")

# Index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # Extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Extract data for additional visualization (e.g., message lengths)
    message_lengths = df['message'].apply(len)

    # Create additional visualization
    

    # Create visuals
    graphs = [
        {
            'data': [
                PlotlyBar(
                    x =list(df[df.columns[8:]].columns),
                    y = df[df.columns[8:]].sum()
                    #x=genre_names,
                    #y=genre_counts
                )
            ],
            'layout': {
                'title': 'Categories',
                'yaxis': {
                    'title': "Message Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                PlotlyBar(
                    x=df['genre'].value_counts().index,
                    y=df['genre'].value_counts().values,
                    marker=dict(color='orange'),  # Customize the color of bars
                    opacity=0.75  # Adjust the opacity of bars
                )
            ],
            'layout': {
                'title': 'Message Genre Distribution',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
    'data': [
        PlotlyBar(
                    x=genre_names,
                    y=genre_counts
        )
    ],
    'layout': {
        'title': 'Message Genre Distribution',
        'xaxis': {
            'title': "Count"
        },
        'yaxis': {
            'title': "Genre"
        }
    }

        }
    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Web page that handles user query and displays model results
@app.route('/go')
def go():
    # Save user input in query
    query = request.args.get('query', '')

    # Use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()