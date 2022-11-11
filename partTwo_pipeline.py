import nltk as nltk
import numpy as np
import sklearn
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

moviedir = r'.\movie_reviews'

# loading all files.
movie = load_files(moviedir, shuffle=True)

docs_train, docs_test, y_train, y_test = train_test_split(movie.data, movie.target,
                                                          test_size=0.20, random_state=12)

text_clf = Pipeline([
 ('vect', CountVectorizer()),
 ('tfidf', TfidfTransformer()),
 ('clf', MultinomialNB()),
])

text_clf.fit(docs_train, y_train)

parameters = {
 'vect__ngram_range': [(1, 1), (1, 2)],
 'tfidf__use_idf': (True, False),
 'clf__alpha': (1e-1, 1e-2),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)

gs_clf = gs_clf.fit(docs_train, y_train)

#print(docs_train[gs_clf.predict(['God is love'])[0]])

print('Gridsearch best score', gs_clf.best_score_)

print('Best params')
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

# Predict the Test set results, find accuracy
predicted = gs_clf.predict(docs_test)
print("Accuracy", np.mean(predicted == y_test))


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predicted)
print('Confusion matrix:', cm)

# very short and fake movie reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride',
               'Steven Seagal was terrible', 'Steven Seagal shone through.',
               'This was certainly a good movie', 'Two thumbs up', 'I fell asleep halfway through',
               "We can't wait for the sequel!!!", '!', '?', 'I cannot recommend this highly enough',
               'instant classic.', 'Meryl Streep was amazing. His performance was Oscar-worthy.',
               'I am speechless', 'Steven Seagal', 'Meryl Streep', 'James Corden', 'Arnold Schwarzenegger',
               'Nicolas Cage', 'Sylvester Stallone', 'Sharknado was amazing',
               ]
thebatmanreview = ["""Everything about this movie is trying too hard - the over dramatic score, the long shots on 
characters faces, the overacting, the complex crime story - it all feels like it's trying to get an Oscar in every 
moment. It's overly long, drawn out, and the story feels like a generic crime saga that has the Batman universe 
shoehorned into it. This movie is not a masterpiece, but it spends a lot of effort making you think it is!"""]

# have classifier make a prediction
pred = gs_clf.predict(reviews_new)
# print out results
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))

# have classifier make a prediction
pred = gs_clf.predict(thebatmanreview)
# print out results
for review, category in zip(thebatmanreview, pred):
    print('%r => %s' % (review, movie.target_names[category]))
