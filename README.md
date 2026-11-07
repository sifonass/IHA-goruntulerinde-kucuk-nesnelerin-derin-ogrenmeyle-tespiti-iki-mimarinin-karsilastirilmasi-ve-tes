# İHA Görüntülerinde Küçük Nesne Tespiti ve Doğal Dilde Özetleme

Bu proje; İHA tarafından alınan elektro-optik (EO) ve termal görüntülerdeki küçük nesnelerin (insan, araç, drone vb.) derin öğrenme mimarileriyle tespit edilmesini, SAHI dilimleme yaklaşımıyla küçük hedef başarımının artırılmasını ve tespit sonuçlarının kural tabanlı bir raporlama modülüyle doğal dil durum özetlerine dönüştürülmesini kapsar.

## Sistem Akış Şeması

```mermaid
graph TD
    A[GİRDİ: Termal İHA Görseli] --> B[Görüntü Ön İşleme & SAHI Dilimleme]
    B --> C[YOLOv8n Çıkarım Motoru]
    C --> D[Tespit Nesneleri & Bounding Box]
    D --> E[Kural Tabanlı Doğal Dil Raporlama Modülü]
    E --> F[ÇIKTI 1: Etiketli Görsel Çıktı]
    E --> G[ÇIKTI 2: Doğal Dil Görev Durum Raporu]


## 1. Veri Seti ve Kümeler

Aşağıdaki tabloda birleştirilmiş İHA veri setinin (`merged_uav_dataset`) train, val ve test kümelerine göre görüntü ve etiket dağılımları yer almaktadır:

| Küme (Split) | Görüntü Sayısı | Etiket (.txt) Sayısı | Toplam Kutu (Instances) | Küçük Kutu (<32x32 px) | Genel Küçük Oranı (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 22338 | 22338 | 72583 | 15517 | %21.38 |
| **Val** | 2789 | 2789 | 9117 | 2182 | %23.93 |
| **Test** | 2798 | 2798 | 8748 | 1732 | %19.80 |

---

## 2. Kümeler Arası Veri Sızıntısı (Data Leakage) Kontrolü

Veri setindeki sızıntı durumunu tespit etmek amacıyla gerçekleştirilen MD5 (birebir eşleşme) ve dHash (algısal benzerlik) analizi sonuçları:

* **Birebir Aynı (MD5) Görüntü Çiftleri:** Toplam **2** adet sızıntı tespit edilmiştir (Örn: `03_llvip__040230__61896b05c9.png` <==> `03_llvip__040179__63ad052405.png`).
* **Çok Benzer (dHash $\le$ 5) Görüntü Çiftleri:** Toplam **453.198** adet yakınlık/çakışma tespit edilmiştir.

> **Önemli Not:** Yüksek miktardaki dHash yakınlığı, video dizisi bazlı (sequence-based) bölme yapılmadığını ve ardışık kare sızıntısı olduğunu göstermektedir. Adil ve güvenilir bir akademik değerlendirme için kümelerin uçuş/dizi ID'sine göre yeniden ayrılması gerekmektedir.

---

## 3. Test Sonuçları (Küme: TEST)

Modelin `TEST` kümesi üzerindeki değerlendirme sonuçları aşağıda tablolanmıştır.

### Genel ve Küçük Nesne Başarım Tablosu

| Metrik | IoU Eşiği / Alan Tanımı | Değer |
| :--- | :--- | :---: |
| **mAP@50 (Genel)** | IoU = 0.50 | **0.9366** |
| **mAP@50-95 (Genel)** | IoU = 0.50:0.95 | **0.7240** |
| **AP_small (<32x32 px)** | IoU = 0.50:0.95 (COCO area = small) | **0.5550** |

### Sınıf Bazlı mAP@50 Tablosu

| Sınıf ID | Sınıf Adı (Class Name) | Örnek Sayısı (Instances) | mAP@50 |
| :---: | :--- | :---: | :---: |
| 0 | person | 5930 | 0.9719 |
| 1 | car | 1694 | 0.9709 |
| 2 | bicycle | 378 | 0.8343 |
| 3 | other_vehicle | 10 | 0.8190 |
| 4 | drone | 367 | 0.9827 |
| 5 | mine | 182 | 0.9944 |
| 6 | gun | 187 | 0.9829 |

---

## 4. Doğal Dil Raporlama Modülü

Nesne tespit modelinden alınan ham çıktıları operatör dostu Türkçe durum raporuna dönüştüren modüldür.

* **Kodun Bulunduğu Dosya:** `report_generator.py`
* **Girdi Formatı:** Sınıf adları ve tespit adedi içeren tespit listesi (Örn: `{'person': 2, 'car': 1}`)
* **Çıktı Formatı:** Doğal dilde rapor cümlesi (Örn: `"Gözlem sahasında 2 adet person ve 1 adet car tespit edilmiştir."`)

---

## 5. Sıradaki Adım

1. Veri setindeki sızıntıları önlemek amacıyla kareleri uçuş/video bazlı (sequence ID) olarak yeniden `train/val/test` kümelerine ayırmak.
2. Temizlenmiş veri kümesi üzerinde **ikinci nesne tespit mimarisini** (örneğin DETR veya RT-DETR) aynı test kümesinde eğitip performansını mevcut YOLO mimarisi ile kıyaslamak.
