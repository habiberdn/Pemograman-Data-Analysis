import pandas as pd 

data = {
    'Nama' : ['Ade','Gitandi', 'Rizky','Yoppi','Benny','Ipul','Raka',
'Andri','','Adit'],
    'Umur' : [39,37,38,39,38,36,39,37,37,39],
    'Alamat' : ['Cengkareng','Tanggerang','Cengkareng','','Serpong','Meruya','Tanggerang','Cilegon','Bekasi','Bekasi']
}

df = pd.DataFrame(data)

print(df)

df.to_csv('data.csv',index=False)