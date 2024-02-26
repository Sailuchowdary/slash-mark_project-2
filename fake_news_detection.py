import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Reading the dataset
df = pd.read_csv('news.csv')
# Getting dimensions of the dataset
df.shape

# Getting first 5 rows of the dataset
df.head()

# Here, there's a column named 'Unnamed: 0', it refers to the article ID to which the given data belongs to
# This column actually has no real use for our model, so let's drop it
df.drop('Unnamed: 0', axis = 1, inplace = True)

df.head()

# Now, let's check for null values in the dataset
df.isnull().sum()

# Since the are no null values, we can proceed further
# Now, let's assume we have an article with very few words
# If we include such smaller text articles into our model,
# we could easily run into accuracy problems with our model.
# So let's handle them

# First, let's add a column 'length' to our DataFrame
# This column indicates the length of the 'text' it corresponds to
df['length'] = [len(str(text)) for text in df['text']]

df.head()

# Let's assume a threshold of length 50, we'll drop any article with text less than 50 characters
print("Number of articles which have less than 50 characters:", len(df[df['length'] < 50]))

# As we saw above, we have 45 rows with text length less than 50. Let's drop them from the table
df.drop(df['text'][df['length'] < 50].index, axis = 0, inplace = True)
# The dimensions of the dataset after dropping them
df.shape

# Now, let's cross-verify if the DataFrame has any articles with less than 50 characters in it's text
print("Number of articles which have less than 50 characters:", len(df[df['length'] < 50]))

# Let's visualize the number of Fake and Real news
plt.figure(figsize = (20, 5), dpi = 96)
plt.title("Counts of labels")
sns.countplot(x = 'label', data = df, color = 'skyblue')
plt.show()

# Let's visualize the distribution of the 'length' column
plt.figure(figsize = (20, 5), dpi = 96)
plt.title("Distribution of the lengths of texts")
plt.hist(df['length'], bins = 30, color = 'green', edgecolor = 'red')
plt.show()

# Now, let's drop the length column from the dataframe as it's of no actual use in prediction model
df.drop('length', axis = 1, inplace = True)

df.head()

# Let's get to building the model now
# Feature and target datasets
X = df['text']
y = df['label']

# Splitting the datasets into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

# Initializing a TfidfVectorizer
tfidf = TfidfVectorizer(stop_words = 'english', max_df = 0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)

# Initializing a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter = 50)
pac.fit(tfidf_train, y_train)

# Predicting on the train dataset and calculating accuracy
y_train_pred = pac.predict(tfidf_train)
print(f'Accuracy on the train dataset: {round(accuracy_score(y_train, y_train_pred) * 100, 2)}%')

# Confusion matrix on train dataset prediction
confusion_matrix(y_train, y_train_pred)

# Classification report of the train dataset prediction
print(classification_report(y_train, y_train_pred))

# Predicting on the test dataset and calculating accuracy
y_test_pred = pac.predict(tfidf_test)
print(f'Accuracy on the test dataset: {round(accuracy_score(y_test, y_test_pred) * 100, 2)}%')

# Confusion matrix on test dataset prediction
confusion_matrix(y_test, y_test_pred)

# Classification report of the test dataset prediction
print(classification_report(y_test, y_test_pred))

# Let's also check predictions with some new data that we manually enter
# You will be asked to enter a random text from a news article as input,
# Then you will be shown if the new is REAL of FAKE
def fake_news_detector():
  random_news = [input(f"{'='*80}\nEnter the news here: ")]
  tfidf_random_news = tfidf.transform(random_news)
  
  random_pred = pac.predict(tfidf_random_news)
  print(f'\nThe news you entered is: {random_pred[0]}')

while True:
  fake_news_detector()

  ask_user = input("\nDo you want to check another news that you have? Choose y/n: ").lower()

  print()
  if (ask_user == "n"):
    break
