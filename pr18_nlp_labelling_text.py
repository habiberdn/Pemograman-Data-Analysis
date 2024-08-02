import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
import re

#Fungsi untuk membac dataset CSV
def load_data(file_path):
    return pd.read_csv(file_path) 

def replace_enter_with_space(text):
    #Replace newline characters with a space
    return text.replace('\n', ' ')

def titik_jadi_koma(text):
    #Replace newline characters with a space
    return text.replace('.', ',')

def titikdua_tambah_spasi(text):
    #Replace newline characters with a space
    return text.replace(':', ': ')

def replace_words(text):
    #Baca file CSV
    dfkamus = pd.read_csv("pr18_kamus.csv")
    #Pastikan CSV memiliki kolom 'old_word' dan 'new_word'
    if 'kata_lama' not in dfkamus.columns or 'kata_baru' not in dfkamus.columns:
        print("CSV harus memiliki kolom 'kata_lama' dan 'kata_baru'")
        return text
    #Ganti setiap kata lama dengan kata baru
    for _, row in dfkamus.iterrows():
        kata_lama = row['kata_lama']
        kata_baru = row['kata_baru']
        text = text.replace(kata_lama, kata_baru)
    return text

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
    
def remove_emoji(string):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642" 
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def proses_preprocessed_text(text):
    text = replace_enter_with_space(text)
    text = titik_jadi_koma(text)
    text = titikdua_tambah_spasi(text)
    text = replace_words(text)
    text = remove_emoji(text)

    return text

def proses_translate_text_en(text):
    text = translate_text(text)
    return text

def proses_labeling(text):
    text = analyze_sentiment(text)

    return text

#Main Function
def main():
    #Path to CSV file
    file_path = 'pr18_dataset_sentimen_10.csv'
    #load data
    df = load_data(file_path)
 
    # pastikan kolom 'komentar ada dalam dataset'
    if 'komentar' not in df.columns:
        raise ValueError("Kolom komentar tidak ditemukan.")
    df['komentar'] = df['komentar'].apply(proses_preprocessed_text)
    df['hasil_translate'] = df['komentar'].apply(proses_translate_text_en)
    df['label'] = df['hasil_translate'].apply(proses_labeling)
    #simpan hasil preprocessing ke file baru
    df.to_csv('pr18_dataset_sentimen_10_cleaned.csv', index=False)
    print("Proses Selesai.")

if __name__ == "__main__":
    main()