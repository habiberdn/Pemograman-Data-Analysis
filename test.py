import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# nltk.download('punkt')

def replace_words(text):
    # Baca file CSV
    dfkamus = pd.read_csv("pr18_kamus.csv")
    
    # Pastikan CSV memiliki kolom 'kata_lama' dan 'kata_baru'
    if 'kata_lama' not in dfkamus.columns or 'kata_baru' not in dfkamus.columns:
        print("CSV harus memiliki kolom 'kata_lama' dan 'kata_baru'")
        return text

    # Tokenize teks asli
    tokens = word_tokenize(" ".join(text))

    # Buat dictionary dari dfkamus
    kamus_dict = {row['kata_lama']: row['kata_baru'] for _, row in dfkamus.iterrows()}

    # Ganti setiap kata lama dengan kata baru, hanya jika kata tersebut ada di kamus
    replaced_tokens = [kamus_dict[token] if token in kamus_dict else token for token in tokens]
    
    return replaced_tokens
# Contoh teks
text = "ngrnjgryg yg"
words = text.split(' ')
# Panggil fungsi replace_words
y = replace_words(words)
print(y)