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
