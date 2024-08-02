import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import matplotlib as mpl

mpl.is_interactive()
plt.ion()
data = pd.read_csv('data.csv')

print("Lima baris pertama : ")
print(data.head())

print("\n Informasi DataFrame :")
print(data.info())

data['Kategori Usia'] = pd.cut(data['Umur'], bins=[0, 30, 40, 50, np.inf], labels=['<30', '30-40', '40-50', '50+'])
count_usia = data['Kategori Usia'].value_counts()

plt.figure(figsize=(8,6))
sns.barplot(x=count_usia.index, y=count_usia.values)
plt.figure(figsize=(8,6))
sns.barplot(x=rata_rata_usia_per_kota.index, y=rata_rata_usia_per_kota.values)   
plt.title('Rata rata usia per alamat')
plt.xlabel('Alamat')
plt.ylabel('Rata Rata Usia')
plt.xticks(rotation=45)
plt.show()
plt.title('Jumlah Individu dalam Setiap Kategori Usia')
plt.xlabel('Kategori Usia')
plt.ylabel('Jumlah Individu')
plt.show()

rata_rata_usia_per_kota = data.groupby('Alamat')['Umur'].mean()


