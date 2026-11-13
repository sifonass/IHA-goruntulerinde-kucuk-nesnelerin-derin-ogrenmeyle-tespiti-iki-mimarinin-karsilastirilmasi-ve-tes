# Elektro-Optik ve Termal İHA Görüntülerinde Küçük Hedeflerin Derin Öğrenme ile Tespiti ve Doğal Dilde Raporlanması

Bu proje; İnsansız Hava Araçlarından (İHA) alınan elektro-optik (EO) ve termal görüntülerde küçük hedeflerin tespit edilmesini, sınıflandırılmasını ve elde edilen sonuçların operasyonel durum raporu olarak doğal dilde sunulmasını amaçlamaktadır.

---

## Sistem Mimarisi ve İş Akışı

<pre>
┌───────────────────────────────────────────────────────────────────────┐
│            Birleştirilmiş İHA Veri Seti (EO ve Termal)                │
└───────────────────────────────────┬───────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Veri Sızıntısı Kontrolü ve Sequence Bölme (MD5 / Hamming / NCC)      │
│  Bölme Parmak İzi: da43dd521edebb0d                                   │
└───────────────────┬───────────────────────────────────┬───────────────┘
                    │                                   │
                    ▼                                   ▼
      ┌───────────────────────────┐       ┌───────────────────────────┐
      │   Eğitim & Doğrulama      │       │   Bağımsız Test Kümesi    │
      │   (Train & Val)           │       │   (Test: 2974 Görüntü)    │
      └─────────────┬─────────────┘       └─────────────┬─────────────┘
                    │                                   │
          ┌─────────┴─────────┐                         │
          │                   │                         │
          ▼                   ▼                         │
  ┌───────────────┐   ┌───────────────┐                 │
  │ 1. Mimari:    │   │ 2. Mimari:    │                 │
  │ YOLOv8 (CNN)  │   │ RT-DETR (ViT) │                 │
  └───────┬───────┘   └───────┬───────┘                 │
          │                   │                         │
          └─────────┬─────────┘                         │
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Test Değerlendirmesi ve Kıyaslama (mAP@50, mAP@50-95, AP_small)      │
└───────────────────┬───────────────────────────────────┬───────────────┘
                    │                                   │
                    ▼                                   ▼
      ┌───────────────────────────┐       ┌───────────────────────────┐
      │ Performans Metrikleri     │       │ Tespit Edilen Sınıflar    │
      │ (metrics_test.csv)        │       │ ['person', 'car', ...]    │
      └───────────────────────────┘       └─────────────┬─────────────┘
                                                        │
                                                        ▼
                                          ┌───────────────────────────┐
                                          │ Doğal Dil Modülü          │
                                          │ (report_generator.py)     │
                                          └─────────────┬─────────────┘
                                                        │
                                                        ▼
                                          ┌───────────────────────────┐
                                          │ Türkçe Durum Raporu       │
                                          │ (20 Örnek Çıktı)          │
                                          └───────────────────────────┘
</pre>

---

## 1. Veri Seti ve Kümeler

Aşağıdaki tabloda, kümeler arası veri sızıntısını (data leakage) önlemek amacıyla uçuş/dizi (`sequence`) bazlı olarak yeniden ayrılan birleştirilmiş İHA veri setinin (`merged_uav_dataset`) test kümesine ait görüntü ve etiket dağılımları yer almaktadır:

| Küme (Split) | Görüntü Sayısı | Etiket (`.txt`) Sayısı | Toplam Kutu (Instances) | Küçük Kutu (`<32x32 px`) | Genel Küçük Oranı (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Test** | 2974 | 2974 (97 arka plan) | 9611 | 2295 | %23.88 |

---

## 2. Kümeler Arası Veri Sızıntısı (Data Leakage) Kontrolü ve Çözümü

İlk aşamada yapılan MD5 (birebir eşleşme) ve dHash (algısal benzerlik) kontrollerinde tespit edilen sızıntılar giderilmiş, veri seti yeniden yapılandırılmış ve son doğrulama testleri tamamlanmıştır:

* **Önceki Durum:** Kümeler rastgele bölündüğü için 2 adet birebir aynı (MD5) ve 453.198 adet çok benzer (`dHash <= 5`) ardışık video karesi çakışması tespit edilmişti.
* **Uygulanan Çözüm:** Adil ve akademik olarak güvenilir bir değerlendirme sağlamak amacıyla tüm veri seti uçuş/video dizisi (`sequence ID`) bazlı olarak yeniden `train`, `val` ve `test` kümelerine ayrılmıştır.
* **Son Doğrulama Sonuçları (MD5 / Hamming / NCC):**
  * **Birden Fazla Kümeye Dağılmış Grup:** `0`
  * **Kümeler Arası Birebir Kopya (MD5):** `0`
  * **Kümeler Arası Yakın Kopya:** `0`
  * **Farklı Kümelerde Aynı Dosya Adı:** `0`
  * **Küme Çiftleri Arası Maksimum Benzerlik (NCC, Ham<=10):**
    * `val -> train`: `0.995`
    * `test -> train`: `1.000`
    * `test -> val`: `0.976`
  * **Doğrulama Çıktısı:** `SIZINTI YOK - train/val/test ayrık` (**Bölme Parmak İzi / Split Fingerprint:** `da43dd521edebb0d`)

---

## 3. Test Sonuçları (Küme: TEST — Sızıntısız)

Sızıntılardan arındırılmış 2974 görüntülük (9611 nesne, 2295 küçük nesne) güncel **TEST** kümesi üzerinde iki farklı derin öğrenme mimarisi (YOLOv8 ve RT-DETR) ayrı ayrı değerlendirilmiştir.

### 3.1. YOLOv8 Test Sonuçları

#### Genel ve Küçük Nesne Başarım Tablosu (YOLOv8)

| Metrik | IoU Eşiği / Alan Tanımı | Değer |
| :--- | :--- | :---: |
| **mAP@50 (Genel)** | IoU = 0.50 | 0.8970 |
| **mAP@50-95 (Genel)** | IoU = 0.50:0.95 | 0.6221 |
| **AP_small@50 (`<32x32 px`)** | IoU = 0.50 (COCO area <= 1024px) | 0.7631 |
| **AP_small@50-95 (`<32x32 px`)** | IoU = 0.50:0.95 (COCO area <= 1024px) | 0.4491 |

#### Sınıf Bazlı Başarım Tablosu (YOLOv8)

| Sınıf ID | Sınıf Adı (Class Name) | Örnek Sayısı (Instances) | Küçük Nesne Sayısı (`<32x32 px`) | mAP@50 | mAP@50-95 | AP_small@50-95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | `person` | 6605 | 1568 | 0.9677 | 0.6617 | 0.5166 |
| 1 | `car` | 1906 | 44 | 0.9889 | 0.9062 | 0.5523 |
| 2 | `bicycle` | 515 | 350 | 0.8811 | 0.5175 | 0.4465 |
| 3 | `other_vehicle` | 19 | 1 | 0.5917 | 0.3568 | 0.0000 |
| 4 | `drone` | 383 | 253 | 0.9575 | 0.6589 | 0.5795 |
| 5 | `mine` | 183 | 79 | 0.9949 | 0.6316 | 0.5999 |

> **YOLOv8 Değerlendirme Notu:** Sızıntı öncesi (ardışık kare ezberli) dağılımda `%93.66` olan `mAP@50` değeri, sızıntılar tamamen temizlendikten sonra gerçek genelleme performansını yansıtarak `%89.70` olarak gerçekleşmiştir. Ana sınıfların (`person`, `car`, `drone`, `mine`) tamamı `%95.7–%99.5` bandında yüksek başarım sergilerken, `other_vehicle` sınıfındaki düşük skor test kümesindeki örnek azlığından (`n=19`) kaynaklanmaktadır.

---

### 3.2. RT-DETR Test Sonuçları

#### Genel ve Küçük Nesne Başarım Tablosu (RT-DETR)

| Metrik | IoU Eşiği / Alan Tanımı | Değer |
| :--- | :--- | :---: |
| **mAP@50 (Genel)** | IoU = 0.50 | **0.9119** |
| **mAP@50-95 (Genel)** | IoU = 0.50:0.95 | **0.6684** |
| **AP_small@50 (`<32x32 px`)** | IoU = 0.50 (COCO area <= 1024px) | **0.7964** |
| **AP_small@50-95 (`<32x32 px`)** | IoU = 0.50:0.95 (COCO area <= 1024px) | **0.5306** |

#### Sınıf Bazlı Başarım Tablosu (RT-DETR)

| Sınıf ID | Sınıf Adı (Class Name) | Örnek Sayısı (Instances) | Küçük Nesne Sayısı (`<32x32 px`) | mAP@50 | mAP@50-95 | AP_small@50-95 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | `person` | 6605 | 1568 | 0.9803 | - | 0.5531 |
| 1 | `car` | 1906 | 44 | 0.9885 | - | 0.5543 |
| 2 | `bicycle` | 515 | 350 | 0.9201 | - | 0.5471 |
| 3 | `other_vehicle` | 19 | 1 | 0.6273 | - | 0.0667 |
| 4 | `drone` | 383 | 253 | 0.9627 | - | 0.6164 |
| 5 | `mine` | 183 | 79 | 0.9921 | - | 0.8462 |

> **RT-DETR Değerlendirme Notu:** Transformer tabanlı öz-dikkat (self-attention) mimarisine sahip RT-DETR modeli, sızıntısız bağımsız test kümesinde `%91.19` genel `mAP@50` ve `%53.06` `AP_small@50-95` başarımına ulaşmıştır. Özellikle küçük hedeflerin yoğun olduğu `mine` (`AP_small@50-95 = 0.8462`), `drone` (`0.6164`) ve `bicycle` (`0.5471`) sınıflarında güçlü konumlandırma hassasiyeti sergilemiştir.

---

### 3.3. YOLOv8 vs. RT-DETR Karşılaştırma Tablosu

Aynı veri bölmesi (`da43dd521edebb0d`) ve aynı test kümesi (`2974 görüntü`, `9611 nesne`) üzerinde iki mimarinin karşılaştırmalı performans özeti aşağıdadır:

| Metrik / Kriter | YOLOv8 (CNN) | RT-DETR (Transformer) | Fark (RT-DETR - YOLOv8) |
| :--- | :---: | :---: | :---: |
| **mAP@50 (Genel)** | 0.8970 | **0.9119** | `+0.0149` (+%1.49) |
| **mAP@50-95 (Genel)** | 0.6221 | **0.6684** | `+0.0463` (+%4.63) |
| **AP_small@50 (`<32x32 px`)** | 0.7631 | **0.7964** | `+0.0333` (+%3.33) |
| **AP_small@50-95 (`<32x32 px`)** | 0.4491 | **0.5306** | `+0.0815` (+%8.15) |
| **`person` mAP@50 / AP_small@50-95** | 0.9677 / 0.5166 | **0.9803 / 0.5531** | `+0.0126` / `+0.0365` |
| **`car` mAP@50 / AP_small@50-95** | **0.9889** / 0.5523 | 0.9885 / **0.5543** | `-0.0004` / `+0.0020` |
| **`bicycle` mAP@50 / AP_small@50-95** | 0.8811 / 0.4465 | **0.9201 / 0.5471** | `+0.0390` / `+0.1006` |
| **`other_vehicle` mAP@50 / AP_small@50-95** | 0.5917 / 0.0000 | **0.6273 / 0.0667** | `+0.0356` / `+0.0667` |
| **`drone` mAP@50 / AP_small@50-95** | 0.9575 / 0.5795 | **0.9627 / 0.6164** | `+0.0052` / `+0.0369` |
| **`mine` mAP@50 / AP_small@50-95** | **0.9949** / 0.5999 | 0.9921 / **0.8462** | `-0.0028` / `+0.2463` |

> **Karşılaştırmalı Sonuç:** RT-DETR, özellikle `<32x32 px` boyutundaki küçük hedeflerde (`AP_small@50-95` metriğinde **+%8.15** artış) ve genel kutu hizalama hassasiyetinde (`mAP@50-95` metriğinde **+%4.63** artış) CNN tabanlı YOLOv8 mimarisine kıyasla belirgin üstünlük sağlamıştır.

---

## 4. Doğal Dil Raporlama Modülü ve 20 Örnek Çıktı

Nesne tespit modelinden alınan ham çıktıları operatör dostu Türkçe durum raporuna dönüştüren modüldür:

* **Kodun Bulunduğu Dosya:** `report_generator.py`
* **Girdi Formatı:** Sınıf adları ve tespit adedi içeren tespit listesi (Örn: `['person', 'person', 'car']`)
* **Çıktı Formatı:** Doğal dilde rapor cümlesi (Örn: `"Gözlem sahasında 2 adet person ve 1 adet car tespit edilmiştir."`)

### Test Kümesinden Seçilen 20 Örnek Tespit ve Durum Raporları

*(Görsellerde **yeşil kutular** gerçek etiketleri (GT), **kırmızı kutular** `conf >= 0.25` eşiğindeki model tahminlerini temsil etmektedir.)*

| Örnek | Tespit Görseli | Doğal Dil Durum Raporu |
| :---: | :---: | :--- |
| **01** | ![01](ornek_ciktilar/01_0001053.jpg) | Gözlem sahasında 6 adet person tespit edilmiştir. |
| **02** | ![02](ornek_ciktilar/02_0001673.jpg) | Gözlem sahasında 4 adet car ve 1 adet person tespit edilmiştir. |
| **03** | ![03](ornek_ciktilar/03_0002483.jpg) | Gözlem sahasında 4 adet car ve 2 adet person tespit edilmiştir. |
| **04** | ![04](ornek_ciktilar/04_0002793.jpg) | Gözlem sahasında 3 adet car ve 2 adet person tespit edilmiştir. |
| **05** | ![05](ornek_ciktilar/05_11_000109.jpg) | Gözlem sahasında 1 adet drone tespit edilmiştir. |
| **06** | ![06](ornek_ciktilar/06_11_000269.jpg) | Gözlem sahasında 1 adet drone tespit edilmiştir. |
| **07** | ![07](ornek_ciktilar/07_13_000264.jpg) | Gözlem sahasında 3 adet drone tespit edilmiştir. |
| **08** | ![08](ornek_ciktilar/08_14_000237.jpg) | Gözlem sahasında 2 adet drone tespit edilmiştir. |
| **09** | ![09](ornek_ciktilar/09_17_000371.jpg) | Gözlem sahasında 2 adet drone tespit edilmiştir. |
| **10** | ![10](ornek_ciktilar/10_23_000253.jpg) | Gözlem sahasında 2 adet drone tespit edilmiştir. |
| **11** | ![11](ornek_ciktilar/11_240418.jpg) | Gözlem sahasında 17 adet person tespit edilmiştir. |
| **12** | ![12](ornek_ciktilar/12_250529.jpg) | Gözlem sahasında 10 adet person tespit edilmiştir. |
| **13** | ![13](ornek_ciktilar/13_250558.jpg) | Gözlem sahasında 4 adet person tespit edilmiştir. |
| **14** | ![14](ornek_ciktilar/14_250596.jpg) | Gözlem sahasında 3 adet person tespit edilmiştir. |
| **15** | ![15](ornek_ciktilar/15_260153.jpg) | Gözlem sahasında 1 adet person tespit edilmiştir. |
| **16** | ![16](ornek_ciktilar/16_260521.jpg) | Gözlem sahasında 1 adet person tespit edilmiştir. |
| **17** | ![17](ornek_ciktilar/17_260676.jpg) | Gözlem sahasında 4 adet person tespit edilmiştir. |
| **18** | ![18](ornek_ciktilar/18_260967.jpg) | Gözlem sahasında 2 adet person tespit edilmiştir. |
| **19** | ![19](ornek_ciktilar/19_270126.jpg) | Gözlem sahasında 2 adet person tespit edilmiştir. |
| **20** | ![20](ornek_ciktilar/20_270296.jpg) | Gözlem sahasında 5 adet person tespit edilmiştir. |
