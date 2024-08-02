import pandas as pd
import re 
import string
import seaborn as sns
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
def load_data(file):
    return pd.read_csv(file)

def clean_text(text):
    text = text.lower()

    text = re.sub(r'http\S+|www\S+|https\S+','',text,flags=re.MULTILINE)

    text = re.sub(r'\d+','',text)
    text = text.translate(str.maketrans('','',string.punctuation))
    return text

def stem_text(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_text = stemmer.stem(text)
    return stemmed_text

def remove_stopward(text):
    stop_words = set(stopwords.words('indonesian'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return  ''.join(filtered_text)


def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopward(text)
    text = stem_text(text)
    return text

def main():
    path = "IF3_22101152630116_Habib_uasdataset_cleaned.csv"
    df = load_data(path)

    if "komentar" not in df.columns or 'label' not in df.columns:
        raise ValueError("Kolom komentar tidak ditemukan")
    
    x_train,x_test,y_train,y_test = train_test_split(df['komentar'].apply(preprocess_text),df['label'],test_size=0.33  ,random_state=42)

    vectorizer = CountVectorizer()
    x_train_vectorizer = vectorizer.fit_transform(x_train)
    x_test_vectorizer = vectorizer.transform(x_test)

    model = MultinomialNB()
    model.fit(x_train_vectorizer,y_train)

    y_pred = model.predict(x_test_vectorizer)

    accuracy = accuracy_score(x_test,y_pred)
    report = classification_report(y_test,y_pred)

    print(f'Accuracy:{accuracy}')
    print('Classification Report')
    print(report)


if __name__ == '__main__':
    main()
