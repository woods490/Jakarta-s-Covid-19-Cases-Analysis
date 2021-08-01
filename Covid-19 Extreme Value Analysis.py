import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pyextremes
from pyextremes import EVA
import scipy.stats as stats

#Memuat Data
data = pd.read_csv(r'C:\Daily_Update_Data_Agregat_Covid-19_Jakarta.csv')
select=["Tanggal Jam", "Positif Harian"]
df = data[select]

#Fungsi deteksi outlier
def detect_outlier(data):
  outliers = []
  threshold = 3
  mean_1 = np.mean(data)
  std_1 = np.std(data)

  for y in data:
    z_score = (y-mean_1)/std_1
    if np.abs(z_score) > threshold:
      outliers.append(y)
    
  return outliers

#Memasukkan kolom 'Positif Harian' pada fungsi deteksi outlier
outlier_datapoints = detect_outlier(df['Positif Harian'])
print('outliers found:')
print(outlier_datapoints)

#Membuat looping untuk menambahkan semua outlier yang terdeteksi pada suatu array
outliers_data = []
for i in range(len(df)):
  for j in outlier_datapoints:
    if df.iloc[i, 1] == j:
      outliers_data.append(df.iloc[i,:])

print()
print('Detail Data Outlier:')
outliers_data = pd.DataFrame(outliers_data)
print(outliers_data)

#Membuat visualisasi Outliers dengan Scatter Plot
x = outliers_data['Tanggal Jam']
y = outliers_data['Positif Harian']
plt.plot(x,y,'ro')
plt.xlabel("Tanggal", labelpad = 30)
plt.ylabel("Outliers", labelpad = 15)
plt.title("Scatter Plot Outliers Positif Harian Jakarta", y=1.012, fontsize=22)
for i_x, i_y in zip(x, y):
    plt.text(i_x, i_y, '({}, {})'.format(i_x, i_y))
plt.show()

#Mengubah index menjadi datetimeindex
datetimeindex = df['Tanggal Jam'].str.replace('/', '-')
df['Tanggal Jam'] = pd.to_datetime(datetimeindex)
df.set_index('Tanggal Jam', inplace=True)

#Menghilangkan Outliers
z_scores = stats.zscore(df)

abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df = df[filtered_entries]

print(df)

#Membuat model Extreme Value Analysis
df1 = df['Positif Harian']
dfseries = pd.Series(df1)
model = EVA(data=dfseries)

#Analisis threshold yang sesuai
pyextremes.plot_parameter_stability(ts=dfseries)
plt.show()

#Memperlihatkan extreme value berdasarkan threshold
model.get_extremes(
    method="POT",
    threshold= 3644.0,
    r='24H',
    )
model.plot_extremes()
plt.show()

#Menjalankan model dan memprediksi extreme value dengan metode Markov chain Monte Carlo
model.fit_model(model='Emcee')
print(model)

model.plot_diagnostic()

plt.show()

summary = model.get_summary(
    return_period=[30, 90, 180, 270, 365],
    )
print("")
print(summary)

#plt.xlabel("Tanggal", labelpad = 10)
#plt.ylabel("Max Positif Harian Selama Satu Bulan", labelpad = 10)
#plt.title("Grafik Max Positif Bulanan Jakarta", y=1.012, fontsize=22)
#plt.show()

#df.plot(kind='line')
#plt.xlabel("Tanggal", labelpad=15)
#plt.ylabel("Positif Harian", labelpad=15)
#plt.title("Grafik Positif Harian Jakarta", y=1.012, fontsize=22)

#plt.show()

