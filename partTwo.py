import nltk as nltk
import sklearn
from sklearn.datasets import load_files

moviedir = r'.\movie_reviews'

# loading all files.
movie = load_files(moviedir, shuffle=True)
print(len(movie.data))

# Split data into training and test sets
from sklearn.model_selection import train_test_split

docs_train, docs_test, y_train, y_test = train_test_split(movie.data, movie.target,
                                                          test_size=0.20, random_state=12)
from sklearn.feature_extraction.text import CountVectorizer

# initialize CountVectorizer
movieVzer = CountVectorizer(min_df=2, tokenizer=nltk.word_tokenize,
                            max_features=3000)  # use top 3000 words only. 78.25% acc.

# fit and tranform using training text
docs_train_counts = movieVzer.fit_transform(docs_train)

# 'screen' is found in the corpus, mapped to index 2290
print(movieVzer.vocabulary_.get('screen'))

# Likewise, Mr. Steven Seagal is present...
print(movieVzer.vocabulary_.get('seagal'))

# huge dimensions! 1,600 documents, 3K unique terms.
print(docs_train_counts.shape)

from sklearn.feature_extraction.text import TfidfTransformer

# Convert raw frequency counts into TF-IDF values
movieTfmer = TfidfTransformer()
docs_train_tfidf = movieTfmer.fit_transform(docs_train_counts)

# Same dimensions, now with tf-idf values instead of raw frequency counts
print(docs_train_tfidf.shape)

# Using the fitted vectorizer and transformer, tranform the test data
docs_test_counts = movieVzer.transform(docs_test)
docs_test_tfidf = movieTfmer.transform(docs_test_counts)

# Now ready to build a classifier.
# We will use Multinominal Naive Bayes as our model
from sklearn.naive_bayes import MultinomialNB

# Train a Multimoda Naive Bayes classifier. Again, we call it "fitting"
clf = MultinomialNB()
clf.fit(docs_train_tfidf, y_train)

# Predict the Test set results, find accuracy
y_pred = clf.predict(docs_test_tfidf)
print(sklearn.metrics.accuracy_score(y_test, y_pred))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

# very short and fake movie reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride',
               'Steven Seagal was terrible', 'Steven Seagal shone through.',
               'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through',
               "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough',
               'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']

reviews_new_counts = movieVzer.transform(reviews_new)  # turn text into count vector
reviews_new_tfidf = movieTfmer.transform(reviews_new_counts)  # turn into tfidf vector

# have classifier make a prediction
pred = clf.predict(reviews_new_tfidf)
# print out results
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))