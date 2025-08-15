# Zaman Serisi Analizine Giriş: Prophet ile Apple Hisse Senedi Fiyat Tahmini

Finans dünyasında, geçmiş fiyat verilerinden yola çıkarak geleceği tahmin etmek oldukça yaygın bir yaklaşımdır. Bu yazıda, Apple (AAPL) hisse senedi fiyatlarını **Facebook Prophet** kütüphanesi ile analiz edip 30 günlük tahmin yapacağız.

---

## 1. Veri Seti ve İlk Bakış

Verimizi `yfinance` kütüphanesi ile çekiyoruz:

```python
import yfinance as yf
df = yf.download('AAPL', start='2018-01-01', end='2023-12-31')
```

Apple’ın son 5 yıldaki fiyat hareketini görselleştirelim:

![AAPL Price Chart](image1.png)

Grafikten görüldüğü üzere Apple, pandemi döneminde düşüş yaşamış ancak ardından güçlü bir yükseliş trendine girmiş.

---

## 2. Trend ve Mevsimsellik Analizi

Zaman serilerinde iki önemli unsur vardır: **Trend** ve **Seasonality (Mevsimsellik)**.

```python
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df['Close'], model='additive', period=252)
result.plot()
```

Bu grafik, fiyatların uzun vadeli artış trendini ve yıl içindeki mevsimsel dalgalanmalarını açıkça ortaya koyuyor.

---

## 3. Prophet ile Tahmin Modeli

Prophet, zaman serisi tahminleri için güçlü ve kolay bir araçtır.

```python
from prophet import Prophet
df_prophet = df.reset_index()[['Date', 'Close']]
df_prophet.columns = ['ds', 'y']

model = Prophet(daily_seasonality=True)
model.fit(df_prophet)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
```

---

## 4. Tahmin Sonuçları

![Prophet Forecast](forecast_chart.png)

İlk grafikte mavi çizgi tahminleri, açık mavi alan ise güven aralığını gösterir.  
İkinci grafikte ise **trend**, **haftalık** ve **yıllık** etkiler ayrı ayrı incelenebilir.

---

## 5. Model Performansı

Tahminin doğruluğunu ölçmek için **MAE** ve **RMSE** metriklerini kullandık:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")
```

Sonuç:  
- **MAE** değeri, modelimizin ortalama ne kadar hata yaptığını gösterir.  
- **RMSE** ise büyük hatalara daha fazla ceza verir ve tahminlerin genel başarısını ölçer.

---

## Sonuç

Bu proje ile:
- Zaman serilerindeki trend ve mevsimsellik yapısını inceledik.
- Prophet ile 30 günlük fiyat tahmini yaptık.
- Modelin doğruluk metriklerini hesapladık.

📌 İleri adımlarda, farklı hisse senetlerini veya kripto para birimlerini analiz ederek model performanslarını karşılaştırabilirsiniz.

---

✍🏻 **Yazar Notu:** Bu yazı, veri bilimi yolculuğumun bir parçası olarak hazırlandı. Siz de kendi veri setlerinizle deneyerek sonuçları yorumlayabilirsiniz.
