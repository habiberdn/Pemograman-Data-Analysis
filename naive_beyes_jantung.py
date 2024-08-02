import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Load data
data = pd.read_csv('pr11_data.csv') # Ganti 'data.csv' dengan nama file CSV Anda
# Pisahkan fitur dan target
X = data.drop('target', axis=1)
y = data['target']
print(y)
# 1. age
# Membuat Grafik age (<30, 40, 35, 40, 45, 50, 55, >60)
# Membuat kolom baru untuk mengkategorikan usia
data['Age'] = pd.cut(data['age'], bins = [0, 30, 35, 40, 45, 50, 55, 60, np.inf],
labels=['<30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60', '60+'])
#Menghitung jumlah individu dalam setiap kategori usia
count_usia = data['Age'].value_counts()
# Visualisasi jumlah individu dalam setiap kategori usia
plt.figure(figsize=(8, 6))
sns.barplot(x=count_usia. index, y=count_usia.values) 
plt.title('Jumlah Pasien sesuai Usia')
plt.xlabel('Usia')
plt.ylabel('Jumlah Pasien')
plt.show()

# 2. sex
# Membuat Grafik sex (0 = perempuan, 1 laki-laki)
# Membuat kolom baru untuk mengkategorikan jenis kelamin 
data['Sex'] = pd.cut (data['sex'], bins=[-0.1, 0.9, np.inf], labels=['Perempuan', 'Laki-Laki'])
# Menghitung jumlah individu dalam setiap kategori jenis kelamin
count_jk = data['Sex'].value_counts()
# Visualisasi jumlah individu dalam setiap kategori jenis kelamin 
plt.figure(figsize=(8, 6))
sns.barplot(x=count_jk. index, y=count_jk.values)
plt.title('Jumlah Pasien sesuai Jenis Kelamin')
plt.xlabel('Jenis Kelamin')
plt.ylabel('Jumlah Pasien')
plt.show()

# Proses KLASIFIKASI dengan NAIVE BAYES
# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Inisialisasi dan latih model Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)
# Prediksi pada set pengujian
y_pred = model.predict(X_test)
# Evaluasi model
print("Akurasi: ", accuracy_score (y_test, y_pred))
print("Hasil Klasifikasi : \n", classification_report (y_test, y_pred))
# Plot confusion matrix
conf_matrix = confusion_matrix (y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
# # 3. Cp

# data['Cp'] = pd.cut (data['cp'], bins=[-0.1, 0.9,1.9,2.9, np.inf], labels=['type 0','type 1', 'type 2','type 3'])
# count_jk = data['Cp'].value_counts()
# plt.figure(figsize=(8, 6))
# sns.barplot(x=count_jk. index, y=count_jk.values)
# plt.title('Jumlah Chess Pain sesuai tipe')
# plt.xlabel('Jenis Type')
# plt.ylabel('Jumlah Pasien')
# plt.show()

# # # 4. Trestbps

# data['Trestbps'] = pd.cut (data['trestbps'], bins = [0, 110, 130, 150, 170, 190, np.inf],
# labels=['<110', '110-130', '130-150', '150-170', '170-190', '190+'])
# count_jk = data['Trestbps'].value_counts()
# plt.figure(figsize=(8, 6))
# sns.barplot(x=count_jk. index, y=count_jk.values)
# plt.title('Jumlah Resting Blood Pressure')
# plt.xlabel('Jenis Jumlah Resting Blood Pressure')
# plt.ylabel('Jumlah Pasien')
# plt.show()

# #5 Chol

# data['Chol'] = pd.cut (data['chol'], bins = [0, 150, 250, 350, 450, 550, np.inf],
# labels=['<150', '150-250', '250-350', '350-450', '450-550', '550+'])
# count_jk = data['Chol'].value_counts()
# plt.figure(figsize=(8, 6))
# sns.barplot(x=count_jk. index, y=count_jk.values)
# plt.title('Jumlah Serum cholestrol')
# plt.xlabel('Jenis Serum cholestrol')
# plt.ylabel('Jumlah Pasien')
# plt.show()

# #6 restecg

# data['Restecg'] = pd.cut (data['restecg'], bins = [-0.1, 0.9,1.9, np.inf],
# labels=['0', '1', '2', ])
# count_jk = data['Restecg'].value_counts()
# plt.figure(figsize=(8, 6))
# sns.barplot(x=count_jk. index, y=count_jk.values)
# plt.title('Jumlah Nilai Resting Electrocardiographic')
# plt.xlabel('Jenis Nilai Resting Electrocardiographic')
# plt.ylabel('Jumlah Pasien')
# plt.show()

# #7 fbs

# data['Fbs'] = pd.cut (data['fbs'], bins = [-0.1, 0.9, np.inf],
# labels=['0', '1' ])
# count_jk = data['Fbs'].value_counts()
# plt.figure(figsize=(8, 6))
# sns.barplot(x=count_jk. index, y=count_jk.values)
# plt.title('Jumlah FBS')
# plt.xlabel('Jenis Fbs')
# plt.ylabel('Jumlah Pasien')
# plt.show()

# #8 Thalach

# data['Thalach'] = pd.cut (data['thalach'], bins = [0, 90,120,150,180, np.inf],
# labels=['<90', '90-120','120-150','150-180','180+' ])
# count_jk = data['Thalach'].value_counts()
# plt.figure(figsize=(8, 6))
# sns.barplot(x=count_jk. index, y=count_jk.values)
# plt.title('Jumlah Thalach')
# plt.xlabel('Jenis Thalach')
# plt.ylabel('Jumlah Pasien')
# plt.show()

# #9 Exang

# data['Exang'] = pd.cut (data['exang'], bins = [-0.1, 0.9, np.inf],
# labels=['No', 'Yes', ])
# count_jk = data['Exang'].value_counts()
# plt.figure(figsize=(8, 6))
# sns.barplot(x=count_jk. index, y=count_jk.values)
# plt.title('Jumlah Exang')
# plt.xlabel('Jenis Exang')
# plt.ylabel('Jumlah Pasien')
# plt.show()

# #10 oldpeak

# data['Oldpeak'] = pd.cut (data['oldpeak'], bins = [-0.1, 1,3,5,np.inf],
# labels=['<1', '1-3','3-5','>5' ])
# count_jk = data['Oldpeak'].value_counts()
# plt.figure(figsize=(8, 6))
# sns.barplot(x=count_jk. index, y=count_jk.values)
# plt.title('Jumlah Oldpeak')
# plt.xlabel('Jenis Oldpeak')
# plt.ylabel('Jumlah Pasien')
# plt.show()

# #11. slope

# data['Slope'] = pd.cut (data['slope'], bins = [-0.1, 0.9,1.9, np.inf],
# labels=['0', '1','2' ])
# count_jk = data['Slope'].value_counts()
# plt.figure(figsize=(8, 6))
# sns.barplot(x=count_jk. index, y=count_jk.values)
# plt.title('Jumlah Slope')
# plt.xlabel('Jenis Slope')
# plt.ylabel('Jumlah Pasien')
# plt.show()

# #12. cd

# data['Ca'] = pd.cut (data['ca'], bins = [-0.1, 0.9,1.9,2.9,3.9, np.inf],
# labels=['0', '1','2','3','4' ])
# count_jk = data['Ca'].value_counts()
# plt.figure(figsize=(8, 6))
# sns.barplot(x=count_jk. index, y=count_jk.values)
# plt.title('Jumlah Ca')
# plt.xlabel('Jenis Ca')
# plt.ylabel('Jumlah Pasien')
# plt.show()

# #13. thal

# data['Thal'] = pd.cut (data['thal'], bins = [-0.1, 0.9,1.9,2.9, np.inf],
# labels=['0', '1','2','3'])
# count_jk = data['Thal'].value_counts()
# plt.figure(figsize=(8, 6))
# sns.barplot(x=count_jk. index, y=count_jk.values)
# plt.title('Jumlah Thal')
# plt.xlabel('Jenis Thal')
# plt.ylabel('Jumlah Pasien')
# plt.show()


