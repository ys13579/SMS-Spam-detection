import pandas as pd
import numpy as np

df = pd.read_table("D:/SMSSpamCollection.txt", header=None)
df.head()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df[0] = le.fit_transform(df[0])

#converting into lowercase
df[1] = df[1].str.lower()

#removing stop words as they dont count + tokenizing
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

df[1]= df[1].apply(lambda x: ' '.join(
    term for term in x.split() if term not in stop_words)
)

porter = nltk.PorterStemmer()
df[1] = df[1].apply(lambda x: ' '.join(
    porter.stem(term) for term in x.split())
)

df[1]= df[1].str.replace(r'[^\w\d\s]', ' ')
df[1]= df[1].str.replace(r'\s+', ' ')
df[1]= df[1].str.replace(r'^\s+|\s+?$', '')

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X_ngrams = vectorizer.fit_transform(df[1])




from sklearn import svm

y= df[0]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_ngrams,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from sklearn import metrics
clf = svm.LinearSVC(loss='hinge')
clf.fit(X_train, y_train)
y_predSVC = clf.predict(X_test)
metrics.f1_score(y_test, y_predSVC)




