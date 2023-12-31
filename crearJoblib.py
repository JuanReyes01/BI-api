# Instalación de librerias
from joblib import dump
import pandas as pd
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
import nltk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.metrics import plot_precision_recall_curve
from sklearn.base import BaseEstimator, ClassifierMixin
from nltk.tokenize import word_tokenize
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay,
    roc_auc_score, precision_score, recall_score, f1_score, accuracy_score,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


from Limpieza import Limpieza

def crearJoblib():
    #Crear el Pipeline con las transformaciones necesarias
    pipeline = Pipeline([
        ('limpieza', Limpieza()),
        ('vectorizer', TfidfVectorizer(tokenizer=word_tokenize, stop_words=stopwords.words("spanish"))),
        ('classifier', RandomForestClassifier(
            n_estimators=300,
            criterion='gini',
            max_depth=100,
            random_state=4

        ))
    ])

    data_t = pd.read_excel('./data/cat_345.xlsx')
    X_train, X_test, y_train, y_test = train_test_split(data_t[["Textos_espanol"]], data_t["sdg"], random_state=1, stratify=data_t['sdg'])
    pipeline.fit(X_train, y_train)

    rutaPipeline =  "pipeline.joblib"
    dump(pipeline, rutaPipeline)