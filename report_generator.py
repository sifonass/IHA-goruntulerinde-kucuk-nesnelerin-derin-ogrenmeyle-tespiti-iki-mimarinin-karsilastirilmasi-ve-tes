from collections import Counter

def generate_report(detections: list) -> str:
    """
    Sınıf adlarını içeren tespit listesini alır (Örn: ['person', 'person', 'car'])
    ve operatör dostu Türkçe durum raporuna dönüştürür.
    """
    if not detections:
        return "Gözlem sahasında herhangi bir nesne tespit edilmemiştir."

    counts = Counter(detections)
    parts = [f"{count} adet {cls_name}" for cls_name, count in counts.items()]

    if len(parts) == 1:
        summary = parts[0]
    else:
        summary = ", ".join(parts[:-1]) + " ve " + parts[-1]

    return f"Gözlem sahasında {summary} tespit edilmiştir."
