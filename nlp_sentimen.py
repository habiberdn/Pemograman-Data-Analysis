from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator

def translate_text(text, dest_language='en'):
    translator = Translator()
    translation = translator.translate(text, dest=dest_language)
    return translation.text

def analyze_sentiment(text):
    #text = thank u very much
    analyzer = SentimentIntensityAnalyzer()
    sentiment_dict = analyzer.polarity_scores(text)

    compound = sentiment_dict['compound']
    if compound >= 0.05:
        return 'Positif'
    elif compound <= -0.05:
        return 'Negatif'
    else:
        return 'Netral'

teks = "terima kasih banyak"

# Terjemahkan teks ke bahasa Inggris
translated_text = translate_text(teks)
print(translated_text)
# print(f"Teks Terjemahan: '{translated_text}'")

# Menganalisis sentimen dari teks yang diterjemahkan
sentimen = analyze_sentiment(translated_text)
print(f"Sentimen dari teks '{teks}' adalah '{sentimen}'.")

