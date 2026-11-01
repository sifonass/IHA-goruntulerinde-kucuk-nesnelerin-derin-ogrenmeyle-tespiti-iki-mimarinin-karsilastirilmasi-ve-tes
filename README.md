# İHA Görüntülerinde Küçük Nesne Tespiti ve Doğal Dilde Özetleme
Bu proje, İHA (İHA/Drone) tarafından alınan elektro-optik (EO) ve termal görüntülerdeki küçük nesnelerin (insan, araç vb.) derin öğrenme mimarileriyle tespit edilmesini ve tespit sonuçlarının doğal dilde anlamlı özetler haline getirilmesini kapsar.
graph TD
    A[GİRDİ: Termal İHA Görseli] --> B[Görüntü Ön İşleme & SAHI Dilimleme]
    B --> C[YOLOv8n Çıkarım Motoru]
    C --> D[Tespit Nesneleri & Bounding Box]
    D --> E[Kural Tabanlı Doğal Dil Raporlama Modülü]
    E --> F[ÇIKTI 1: Etiketli Görsel Çıktı]
    E --> G[ÇIKTI 2: Doğal Dil Görev Durum Raporu]
