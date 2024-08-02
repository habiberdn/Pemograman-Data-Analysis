import pandas as pd
import re
import string
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Fungsi untuk membaca dataset CSV
def load_data(file_path):
    return pd.read_csv(file_path)

# Fungsi untuk membersihkan teks
def clean_text(text):
    # Ubah ke huruf kecil
    text = text.lower()

    # Hilangkan URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Hilangkan karakter spesial dan angka
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('indonesian'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]

    return ' '.join(filtered_text)

# Fungsi untuk stemming
def stem_text(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stemmed_text = stemmer.stem(text)

    return stemmed_text

def preprocess_text(text):
    text = clean_text(text)
    print(text)
    text = remove_stopwords(text)

    text = stem_text(text)


    return text

# Main function
def main():
    # Path to the CSV file
    file_path = 'pr15_komentar_marketplace.csv'

    # Load data
    df = load_data(file_path)

    # Pastikan kolom 'komentar' ada dalam dataset
    if 'komentar' not in df.columns:
        raise ValueError("Kolom 'komentar' tidak ditemukan dalam dataset.")

    # Terapkan preprocessing ke setiap komentar
    df['cleaned_komentar'] = df['komentar'].apply(preprocess_text)

    # Simpan hasil preprocessing ke file baru
    df.to_csv('pr15_cleaned_komentar_marketplace.csv', index=False)
    print("Preprocessing selesai. Disimpan 'pr15_cleaned_komentar_marketplace.csv'.")

if __name__ == "__main__":
    main()