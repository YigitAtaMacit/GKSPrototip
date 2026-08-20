"""KAMERA_INDEKSI'yi bulmak icin yardimci script: her index icin ayri pencere acar, dogru numarayi ayarlar.py'ye yaz, 'q' ile cik."""
import cv2

MAKS_INDEKS = 6  # 0'dan bu sayiya kadar dener (yetersizse artir)

if __name__ == "__main__":
    yakalayicilar = {}
    for i in range(MAKS_INDEKS):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                yakalayicilar[i] = cap
                print(f"[BULUNDU] index={i} calisiyor - pencere acildi.")
            else:
                cap.release()
        else:
            cap.release()

    if not yakalayicilar:
        print("Hicbir kamera bulunamadi. Telefonun Phone Link uzerinden")
        print("baglandigindan ve Windows Kamera uygulamasinda gorundugunden emin ol.")
        raise SystemExit

    print()
    print(f"Toplam {len(yakalayicilar)} kamera bulundu: {sorted(yakalayicilar.keys())}")
    print("Her pencerenin basligindaki index numarasina bak - telefonun")
    print("goruntusunu gordugun pencerenin numarasini ayarlar.py > KAMERA_INDEKSI'ye yaz.")
    print("Cikmak icin herhangi bir pencerede 'q' tusuna bas.")

    while True:
        for i, cap in yakalayicilar.items():
            ok, kare = cap.read()
            if ok:
                cv2.putText(kare, f"KAMERA_INDEKSI = {i}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow(f"kamera {i}", kare)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for cap in yakalayicilar.values():
        cap.release()
    cv2.destroyAllWindows()