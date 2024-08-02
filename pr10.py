
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix 
import matplotlib.pyplot as plt
import seaborn as sns   
# Load data
data = pd.read_csv('pr10_data.csv') # Ganti 'data.csv' dengan nama file CSV Anda
# Pisahkan fitur dan target
X = data.drop('target_column', axis=1) # Ganti 'target_column' dengan nama kolom target
y=  data['target_column']
# Bagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Inisialisasi dan latih model Naive Bayes
model = GaussianNB()
model.fit(X_train,y_train)


# Lakukan prediksi pada data uji 
y_pred = model.predict(X_test)
# Hitung metrik evaluasi
accuracy = accuracy_score (y_test, y_pred)
precision = precision_score (y_test, y_pred) 
recall =  recall_score (y_test, y_pred)
print("Accuracy: ", accuracy)
print("Precision:", precision) 
print("Recall:", recall)
# Buat confusion matrix
conf_matrix = confusion_matrix (y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False) 
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()