import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

# Veriyi yükle
data = pd.read_csv('bitcoin_2010-07-27_2024-04-25.csv')

# Tarih sütunlarını datetime tipine çevir
data['Start'] = pd.to_datetime(data['Start'])
data['End'] = pd.to_datetime(data['End'])

# Yeni zaman serisi özellikleri ekle
data['day_of_week'] = data['Start'].dt.dayofweek
data['month'] = data['Start'].dt.month

# Yeni özellikler türet
data['Price_diff'] = data['High'] - data['Low']

# Özellikler ve hedef
features = ['Open', 'High', 'Low', 'Volume', 'Market Cap', 'day_of_week', 'month', 'Price_diff']
target = 'Close'

# Özellikleri ve hedefi ayır
X = data[features]
y = data[target]

# Polinomiyel özellikler
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Ridge Regresyon modelini oluştur ve eğit
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Tahmin yap
predictions = model.predict(X_test)

# Model performansını değerlendir
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Modelin MAE değeri: ", mae)
print("Modelin RMSE değeri: ", rmse)
