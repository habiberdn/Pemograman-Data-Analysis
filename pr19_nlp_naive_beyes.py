import pandas as pd
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

#fungsi untuk membaca dataset CSV 
def load_data(file_path):
    return pd.read_csv(file_path)

#fungsi untuk memberikan teks
def clean_text(text):
    #ubah ke huruf kecil
    text = text.lower()

    #hilangkan URL 
    text = re.sub(r'http\S+|www\S+|https\S+','',text, flags=re.MULTILINE)   

    #Hilangkan Karakter spesial dan angka
    text = re.sub(r'\d+','',text)
    text = text.translate(str.maketrans('','',string.punctuation))

    return text

#fungsi untuk menghapus stopwords
def remove_stopword(text):
    stop_words = set(stopwords.words('indonesian'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]

    return ' '.join(filtered_text)

#fungsi untuk stemming 
def stem_text(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_text = stemmer.stem(text)   
    print(stemmed_text)
    return stemmed_text

#fungsi untuk preprocessing lengkap
def preprocess_text(text):
    text = clean_text(text)
    text = remove_stopword(text)
    text = stem_text(text)
    return text

#Main function 
def main():
    #path to the CSV file 
    file_path = 'pr18_dataset_sentimen_10_cleaned.csv'

    #load data
    df = load_data(file_path)

    #pastikan kolom 'komentar' dan sentimen' ada dalam dataset 
    if 'komentar' not in df.columns or 'label' not in df.columns:
        raise ValueError("kolom 'komentar' atau 'label' tidak ada dalam dataset.")
    
    #Split dataset menjadi training dan testing set
    x_train, x_test, y_train, y_test = train_test_split(df['komentar'].apply(preprocess_text),df['label'], test_size=0.2, random_state=42)

    #mengubah teks ke fitur menggunakan CountVectorizer
    vectorizer = CountVectorizer()
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)

    #melatih model Naive Bayes
    model = MultinomialNB()
    model.fit(x_train_vectorized, y_train)

    #prediksi pada data testing
    y_pred = model.predict(x_test_vectorized)

    #Evaluasi model
    accuracy = accuracy_score(x_test, y_pred)
    report = classification_report(y_test, y_pred)

    #Print hasil evaluasi 
    print(f'Accuracy:{accuracy}')
    print('Classification Report:')
    print(report)

if __name__ == "__main__":
    main()