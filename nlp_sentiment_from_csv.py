import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator

# Fungsi untuk membaca dataset CSV
def load_data(file_path):
    return pd.read_csv(file_path)

def translate_text(text, dest_language='en'):
    translator = Translator()
    translation = translator.translate(text, dest=dest_language)
    return translation.text

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_dict = analyzer.polarity_scores(text)

    compound = sentiment_dict['compound']
    if compound >= 0.05:
        return 'Positif'
    elif compound <= -0.05:
        return 'Negatif'
    else:
        return 'Netral'

def labeling_text(text):
    text = translate_text(text)
    text = analyze_sentiment(text)
    return text

# Main function
def main():
    # Path to the CSV file
    file_path = 'pr15_komentar_marketplace.csv'

    # Load data
    df = load_data(file_path)

    # Pastikan kolom 'komentar' ada dalam dataset
    if 'komentar' not in df.columns:
        raise ValueError("Kolom 'komentar' tidak ditemukan.")

    df['sentimen'] = df['komentar'].apply(labeling_text)

    # Simpan hasil preprocessing ke file baru
    df.to_csv('pr17_label_komentar_marketplace.csv', index=False)
    print("Proses Labeling Selesai.")

if __name__ == "__main__":
    main()