import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')


#Fungsi untuk membac dataset CSV
def load_data(file_path):
    return pd.read_csv(file_path, on_bad_lines='skip', engine="python") 

def replace_enter_with_space(text):
    #Replace newline characters with a space
    return text.replace('\n', ' ')

def titik_jadi_koma(text):
    #Replace newline characters with a space
    return text.replace('.', ',')

def titikdua_tambah_spasi(text):
    #Replace newline characters with a space
    return text.replace(':', ': ')

# def replace_words(text):
#     #Baca file CSV
#     dfkamus = pd.read_csv("NLP_Kamus.csv")
#     #Pastikan CSV memiliki kolom 'old_word' dan 'new_word'
#     if 'kata_lama' not in dfkamus.columns or 'kata_baru' not in dfkamus.columns:
#         print("CSV harus memiliki kolom 'kata_lama' dan 'kata_baru'")
#         return text
#     #Ganti setiap kata lama dengan kata baru
#     for _, row in dfkamus.iterrows():
#         kata_lama = str(row['kata_lama'])
#         kata_baru = str(row['kata_baru'])
#         text = text.replace(kata_lama, kata_baru)
#     return text

def replace_words(text):
    dfkamus = pd.read_csv("NLP_Kamus.csv")
    
    if 'kata_lama' not in dfkamus.columns or 'kata_baru' not in dfkamus.columns:
        print("CSV harus memiliki kolom 'kata_lama' dan 'kata_baru'")
        return text

    tokens = word_tokenize(" ".join(text))

    kamus_dict = {row['kata_lama']: row['kata_baru'] for _, row in dfkamus.iterrows()}

    replaced_tokens = [kamus_dict[token] if token in kamus_dict else token for token in tokens]
    
    return replaced_tokens



def clean_emoji(text):
    #remove emoji
    return text.clean(text, no_emoji=True)

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
    
#Remove Emojis
def remove_emoji(text):
    emoji_pattern = str(re.compile("["
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
                           "]+", flags=re.UNICODE))
    return emoji_pattern.sub(r'', text)


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
    file_path = 'NLP_dataset.csv'
    #load data
    df = load_data(file_path)
    # pastikan kolom 'komentar ada dalam dataset'
    if 'komentar' not in df.columns:
        raise ValueError("Kolom komentar tidak ditemukan.")
    df['komentar'] = df['komentar'].apply(proses_preprocessed_text)
    df['hasil_translate'] = df['komentar'].apply(proses_translate_text_en)
    df['label'] = df['hasil_translate'].apply(proses_labeling)
    #simpan hasil preprocessing ke file baru
    df.to_csv('NLP_dataset_cleaned_10.csv', index=False)
    print("Proses Selesai.")

if __name__ == "__main__":
    main()  
