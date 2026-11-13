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

## 3. Doğal Dil Raporlama Modülü

Nesne tespit modellerinden alınan ham çıktıları operatör dostu Türkçe durum raporuna dönüştüren modüldür:

* **Kodun Bulunduğu Dosya:** `report_generator.py`
* **Girdi Formatı:** Sınıf adları ve tespit adedi içeren tespit listesi (Örn: `['person', 'person', 'car']`)
* **Çıktı Formatı:** Doğal dilde rapor cümlesi (Örn: `"Gözlem sahasında 2 adet person ve 1 adet car tespit edilmiştir."`)

---

## 4. Test Sonuçları ve 20 Örnek Çıktı (Küme: TEST — Sızıntısız)

*(Tüm örnek görsellerde **yeşil kutular** gerçek etiketleri (GT), **kırmızı kutular** `conf >= 0.25` eşiğindeki model tahminlerini temsil etmektedir.)*

### 4.1. YOLOv8 Test Sonuçları

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

#### YOLOv8 — Test Kümesinden Seçilen 20 Örnek Tespit ve Durum Raporları

| Örnek | Tespit Görseli (YOLOv8) | Doğal Dil Durum Raporu |
| :---: | :---: | :--- |
| **01** | ![01](ornek_ciktilar_yolo/01_02_hituav__1_60_30_0_00090__3f55603120.jpg) | Gözlem sahasında 6 adet person tespit edilmiştir. |
| **02** | ![02](ornek_ciktilar_yolo/02_03_llvip__120179__b620d3d0ed.jpg) | Gözlem sahasında 4 adet car ve 1 adet person tespit edilmiştir. |
| **03** | ![03](ornek_ciktilar_yolo/03_03_llvip__120274__b933da4a1f.jpg) | Gözlem sahasında 4 adet car ve 2 adet person tespit edilmiştir. |
| **04** | ![04](ornek_ciktilar_yolo/04_03_llvip__120326__c7a24636dd.jpg) | Gözlem sahasında 3 adet car ve 2 adet person tespit edilmiştir. |
| **05** | ![05](ornek_ciktilar_yolo/05_01_uav2uav__01724__218eff6b90.jpg) | Gözlem sahasında 1 adet drone tespit edilmiştir. |
| **06** | ![06](ornek_ciktilar_yolo/06_01_uav2uav__02006__50ed4e02dd.jpg) | Gözlem sahasında 1 adet drone tespit edilmiştir. |
| **07** | ![07](ornek_ciktilar_yolo/07_01_uav2uav__02202__c5c7f0f51b.jpg) | Gözlem sahasında 3 adet drone tespit edilmiştir. |
| **08** | ![08](ornek_ciktilar_yolo/08_01_uav2uav__02243__badce0a7db.jpg) | Gözlem sahasında 2 adet drone tespit edilmiştir. |
| **09** | ![09](ornek_ciktilar_yolo/09_01_uav2uav__02244__b8378f0b59.jpg) | Gözlem sahasında 2 adet drone tespit edilmiştir. |
| **10** | ![10](ornek_ciktilar_yolo/10_01_uav2uav__974__8abc3be555.jpg) | Gözlem sahasında 2 adet drone tespit edilmiştir. |
| **11** | ![11](ornek_ciktilar_yolo/11_02_hituav__0_60_40_0_01660__9ec44a4a6c.jpg) | Gözlem sahasında 17 adet person tespit edilmiştir. |
| **12** | ![12](ornek_ciktilar_yolo/12_02_hituav__1_130_60_0_03936__bd07342887.jpg) | Gözlem sahasında 10 adet person tespit edilmiştir. |
| **13** | ![13](ornek_ciktilar_yolo/13_02_hituav__1_60_60_0_00531__c40fdff772.jpg) | Gözlem sahasında 4 adet person tespit edilmiştir. |
| **14** | ![14](ornek_ciktilar_yolo/14_03_llvip__060108__2ae70a5c94.jpg) | Gözlem sahasında 3 adet person tespit edilmiştir. |
| **15** | ![15](ornek_ciktilar_yolo/15_03_llvip__060290__5e24138524.jpg) | Gözlem sahasında 1 adet person tespit edilmiştir. |
| **16** | ![16](ornek_ciktilar_yolo/16_03_llvip__060380__775feba7a4.jpg) | Gözlem sahasında 1 adet person tespit edilmiştir. |
| **17** | ![17](ornek_ciktilar_yolo/17_03_llvip__240352__1e8fc2e778.jpg) | Gözlem sahasında 4 adet person tespit edilmiştir. |
| **18** | ![18](ornek_ciktilar_yolo/18_03_llvip__260510__f3e898ac80.jpg) | Gözlem sahasında 2 adet person tespit edilmiştir. |
| **19** | ![19](ornek_ciktilar_yolo/19_04_mixdataset__002359__52d81915ee.jpg) | Gözlem sahasında 2 adet person tespit edilmiştir. |
| **20** | ![20](ornek_ciktilar_yolo/20_02_hituav__1_60_50_0_00394__18b8ed41d8.jpg) | Gözlem sahasında 5 adet person tespit edilmiştir. |
---

### 4.2. RT-DETR Test Sonuçları

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

#### RT-DETR — Test Kümesinden Seçilen 20 Örnek Tespit ve Durum Raporları

| Örnek | Tespit Görseli (RT-DETR) | Doğal Dil Durum Raporu |
| :---: | :---: | :--- |
| **01** | ![01](ornek_ciktilar_rtdetr/01_02_hituav__1_60_30_0_00090__3f55603120.jpg) | Gözlem sahasında 6 adet person tespit edilmiştir. |
| **02** | ![02](ornek_ciktilar_rtdetr/02_03_llvip__120179__b620d3d0ed.jpg) | Gözlem sahasında 6 adet car ve 2 adet person tespit edilmiştir. |
| **03** | ![03](ornek_ciktilar_rtdetr/03_03_llvip__120274__b933da4a1f.jpg) | Gözlem sahasında 4 adet car ve 6 adet person tespit edilmiştir. |
| **04** | ![04](ornek_ciktilar_rtdetr/04_03_llvip__120326__c7a24636dd.jpg) | Gözlem sahasında 3 adet car ve 3 adet person tespit edilmiştir. |
| **05** | ![05](ornek_ciktilar_rtdetr/05_01_uav2uav__01724__218eff6b90.jpg) | Gözlem sahasında 1 adet drone tespit edilmiştir. |
| **06** | ![06](ornek_ciktilar_rtdetr/06_01_uav2uav__02006__50ed4e02dd.jpg) | Gözlem sahasında 1 adet drone tespit edilmiştir. |
| **07** | ![07](ornek_ciktilar_rtdetr/07_01_uav2uav__02202__c5c7f0f51b.jpg) | Gözlem sahasında 3 adet drone tespit edilmiştir. |
| **08** | ![08](ornek_ciktilar_rtdetr/08_01_uav2uav__02243__badce0a7db.jpg) | Gözlem sahasında 2 adet drone tespit edilmiştir. |
| **09** | ![09](ornek_ciktilar_rtdetr/09_01_uav2uav__02244__b8378f0b59.jpg) | Gözlem sahasında 1 adet drone tespit edilmiştir. |
| **10** | ![10](ornek_ciktilar_rtdetr/10_01_uav2uav__974__8abc3be555.jpg) | Gözlem sahasında 2 adet drone tespit edilmiştir. |
| **11** | ![11](ornek_ciktilar_rtdetr/11_02_hituav__0_60_40_0_01660__9ec44a4a6c.jpg) | Gözlem sahasında 19 adet person tespit edilmiştir. |
| **12** | ![12](ornek_ciktilar_rtdetr/12_02_hituav__1_130_60_0_03936__bd07342887.jpg) | Gözlem sahasında 10 adet person tespit edilmiştir. |
| **13** | ![13](ornek_ciktilar_rtdetr/13_02_hituav__1_60_60_0_00531__c40fdff772.jpg) | Gözlem sahasında 3 adet person tespit edilmiştir. |
| **14** | ![14](ornek_ciktilar_rtdetr/14_03_llvip__060108__2ae70a5c94.jpg) | Gözlem sahasında 3 adet person tespit edilmiştir. |
| **15** | ![15](ornek_ciktilar_rtdetr/15_03_llvip__060290__5e24138524.jpg) | Gözlem sahasında 1 adet person tespit edilmiştir. |
| **16** | ![16](ornek_ciktilar_rtdetr/16_03_llvip__060380__775feba7a4.jpg) | Gözlem sahasında 1 adet person tespit edilmiştir. |
| **17** | ![17](ornek_ciktilar_rtdetr/17_03_llvip__240352__1e8fc2e778.jpg) | Gözlem sahasında 4 adet person tespit edilmiştir. |
| **18** | ![18](ornek_ciktilar_rtdetr/18_03_llvip__260510__f3e898ac80.jpg) | Gözlem sahasında 2 adet person tespit edilmiştir. |
| **19** | ![19](ornek_ciktilar_rtdetr/19_04_mixdataset__002359__52d81915ee.jpg) | Gözlem sahasında 2 adet person tespit edilmiştir. |
| **20** | ![20](ornek_ciktilar_rtdetr/20_02_hituav__1_60_50_0_00394__18b8ed41d8.jpg) | Gözlem sahasında 5 adet person tespit edilmiştir. |

---

## 5. YOLOv8 vs. RT-DETR Karşılaştırmalı Test Sonuçları

Aynı veri bölmesi (`da43dd521edebb0d`) ve aynı bağımsız test kümesi (`2974 görüntü`, `9611 nesne`, `2295 küçük nesne`) üzerinde iki mimarinin karşılaştırmalı performans özeti aşağıdadır:

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

> **Karşılaştırmalı Değerlendirme:** RT-DETR, özellikle `<32x32 px` boyutundaki küçük hedeflerde (`AP_small@50-95` metriğinde **+%8.15** artış) ve genel kutu hizalama hassasiyetinde (`mAP@50-95` metriğinde **+%4.63** artış) CNN tabanlı YOLOv8 mimarisine kıyasla belirgin üstünlük sağlamıştır.
