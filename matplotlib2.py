import matplotlib
matplotlib.use('TkAgg')
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

data = pd.read_csv('data2.csv')
print("Lima Baris Pertama:")
print(data.head())

print('\n Informasi DataFrame : ')
print(data.info())

data['Kategori Usia'] = pd.cut(data['Usia'], bins=[0,30,40,50,np.inf], labels=['<30','30-40','40-50','50+'])

count_usia = data['Kategori Usia'].value_counts()

plt.figure(figsize=(8,6))
sns.barplot(x=count_usia.index, y=count_usia.values, legend=False)
plt.title('Jumlah Individu dalam Setiap Kategori Usia')
plt.xlabel('Kategori Usia')
plt.ylabel('Jumlah Individu')
plt.show()

rata_rata_gaji = data['Gaji'].mean()
data['Gaji_Above_Average'] = np.where(data['Gaji'] > rata_rata_gaji, "Yes", "No")

plt.figure(figsize=(6,4))
sns.countplot(x='Gaji_Above_Average', data=data, palette='coolwarm')   
plt.title('Proporsi Gaji di Atas Rata Rata')
plt.xlabel('Gaji di Atas Rata-Rata')
plt.ylabel('Jumlah Individu')
plt.show()

rata_rata_gaji_per_usia = data.groupby('Kategori Usia')['Gaji'].mean()

plt.figure(figsize=(10,6))
sns.barplot(x=rata_rata_gaji_per_usia.index, y=rata_rata_gaji_per_usia.values, palette='magma')
plt.title('Rata Rata Gaji per Kategori Usia')
plt.xlabel('Kategori Usia')
plt.ylabel('Rata-rata Gaji')
plt.xticks(rotation=45)
plt.show()

korelasi = data[['Usia','Pengeluaran']].corr()

plt.figure(figsize=(6,4))
sns.heatmap(korelasi, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Korelasi Antar Usia dan Pengeluaran")
plt.show()