# Gerekli kütüphaneleri içe aktar
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Veriyi yükle
data = pd.read_csv('bitcoin_2010-07-27_2024-04-25.csv')

# Check if 'Date' column exists
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

# Hareketli Ortalamalar ve RSI hesaplama
data['SMA_30'] = data['Close'].rolling(window=30).mean()
data['SMA_60'] = data['Close'].rolling(window=60).mean()

# Fonksiyon tanımlaması
def compute_rsi(data, window=14):
    diff = data.diff(1)
    gain = (diff.where(diff > 0, 0)).rolling(window=window).mean()
    loss = (-diff.where(diff < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

data['RSI'] = compute_rsi(data['Close'], 14)  # RSI fonksiyonunun tanımı gereklidir

# Eksik verileri temizle
data.dropna(inplace=True)

# Özellikler ve hedef
features = ['Open', 'High', 'Low', 'Volume', 'Market Cap', 'SMA_30', 'SMA_60', 'RSI']
target = 'Close'

# Özellikleri ve hedefi ayır
X = data[features]
y = data[target]

# Veriyi ölçeklendir
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veriyi eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Tahmin yap ve model performansını değerlendir
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Performans metriklerini yazdır
print("MAE:", mae)
print("MSE:", mse)
print("R^2:", r2)

# En son veri noktasını kullanarak 2025 yılı için fiyat tahmini yap
latest_features = data[features].iloc[-1].values.reshape(1, -1)  # Özellikleri al
latest_features_scaled = scaler.transform(latest_features)  # Özellikleri ölçeklendir
price_prediction_2025 = model.predict(latest_features_scaled)  # Tahmini yap

# Tahmin sonucunu yazdır
print("2025 yılı tahmini fiyatı:", price_prediction_2025[0])
