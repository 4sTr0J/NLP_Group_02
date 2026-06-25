import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer

train=pd.read_csv(r'G:\My Drive\NLP project\data\train_data.csv')

vectorizer = TfidfVectorizer(
    max_features=5000, #limit featues to 5000
    ngram_range=(1, 2) #limit upto bigrams
)

x_train = vectorizer.fit_transform(train['code'])

joblib.dump(
    vectorizer, r'G:\My Drive\NLP project\models\tfidf_vectorizer.pkl'
)

# print(x_train.shape)
