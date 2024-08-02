import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory #library utk memfilter kata

text = "Saya memakan makanan ringan di pantai"
tokens = nltk.word_tokenize(text)
print(tokens)

stopwards_id = stopwords.words('indonesian')
filtered_token = [word for word in tokens if word.lower() not in stopwards_id]
# print("Filtered Tokens: ",filtered_token)

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stemmed_token = [stemmer.stem(word) for word in filtered_token]
# print("Stemmed Tokens: ", stemmed_token)