# Sparkify Churn Project: 
# Customer churning: Predicting the exit

### Project Motivation
I took this churn project to apply all my knowlegde thus far with reagrds to machine learning models, EDA and features.. 


This project leverages multiple Python libraries, Pssprak being the core and  including matplotlib for data visualization, pandas for data manipulation, numpy for numerical computations, seaborn for statistical graphics, and scikit-learn for machine learning algorithms. Through these, we perform comprehensive data analysis and build a recommendation model.

Summary
Pyspark 
 Setting Up Spark
Installation: Install Apache Spark and its dependencies. This often includes installing Java and Hadoop.
Environment: Set up the Spark environment on your local machine or cluster. This includes configuring Spark settings and initializing a Spark session.
 Loading Data
Spark Session: Create a Spark session to interact with Spark.

This project involves the following key tasks to predict the churn for customer.

1. Exploratory Data Analysis (EDA)
we conduct a thorough analysis of the provided dataset. This involves examining the shape of the dataset, the number of articles, users, and other key features. The objective is to understand the underlying structure and characteristics of the data.
Exploratory Data Analysis
loading a small subset of the data and doing basic manipulations within Spark. 
Define Churn
create a column Churn to use as the label for the model, using the Cancellation Confirmation events to define the churn, which happen for both paid and free users.

Explore Data
Once you've defined churn, perform some exploratory data analysis to observe the behaviour for users who stayed vs users who churned. You can start by exploring aggregates on these two groups of users, observing how much of a specific action they experienced per a certain time unit or number of songs played.

2. Feature Engineering
Write a script to extract the necessary features from the smaller subset of data.

3. Modelling
the task was to split the full dataset into train, test, and validation sets, thus testing out several of the machine learning methods. 
Evaluating the accuracy of the various models, tuning parameters as necessary, thereby determining the winning model based on test accuracy and report results on the validation set, using F1 score as the metric to optimize.

By combining these techniques, we aim to create a prediction models that is both effective and scalable, leveraging the power of data science to enhance user experience


## Blog Post
I used medium and more info could be found with Code on notebook and others on  [here](https://medium.com/@malomesuret/customer-churning-predicting-the-exit-b2bd5750e9bd)

## Installations


Installations
To run this project, ensure you have the following libraries installed:
pandas
numpy
seaborn
matplotlib.pyplot
scikit-learn
Matplotlib.pyplot, Matplotlib.ticker
Seaborn
Scikit-Learn
- Matplotlib
- Pandas
- Numpy
- Seaborn



## Licensing, Authors,Acknowledgements.

I acknowledge [Kaggle](https://www.kaggle.com/) 
