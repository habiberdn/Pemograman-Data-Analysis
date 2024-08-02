import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator

def load_data(file):
    return pd.read_csv(file)

def replace_enter_with_space(text):
    return text.replace('\n',' ')

def titik_jadi_koma(text):
    return text.replace('.',',')

def titikdua_tambah_spasi(text):
    return text.replace(':',": ")

def replace_words(text):
    dfkamus = pd.read_csv('IF3_22101152630116_Habib_kamuskata.csv')

    if 'kata_lama' not in dfkamus.columns or 'kata_baru' not in dfkamus.columns:
        print('Harus ada kata_lama dan kata_baru')
        return text
    
    for _,row in dfkamus.iterrows():
        kata_lama = row['kata_lama']
        kata_baru = row['kata_baru']
        text = text.replace(kata_lama,kata_baru)
    return text

def translate_text(text,language="en"):
    translator = Translator()
    translation = translator.translate(text,dest=language)
    return translation.text

def analyze_sentimen(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_dict = analyzer.polarity_scores(text)

    compound = sentiment_dict['compound']

    if compound>= 0.05:
        return "Positif"
    elif compound <= -0.05:
        return "Negatif" 
    else:
        return "Netral"
    
def preproces_text(text):
    text = replace_enter_with_space(text)
    text = titik_jadi_koma(text)
    text = titikdua_tambah_spasi(text)
    text = replace_words(text)
    return text

def translate_to_en(text):
    text = translate_text(text)
    return text

def proses_labeling(text):
    text = analyze_sentimen(text)
    return text


def main():
    file = "uas_dataset_komentar.csv"
    df = load_data(file)

    if("komentar") not in df.columns:
        raise ValueError("Kolom komentar tidak ditemukan")

    df['komentar'] = df['komentar'].apply(preproces_text)
    df['hasil_translate'] = df['komentar'].apply(translate_to_en)
    df['label'] = df['hasil_translate'].apply(proses_labeling)

    df.to_csv("IF3_22101152630116_Habib_uasdataset_cleaned.csv",index=False)
    print("selesai")

if __name__ == "__main__":
    main()

