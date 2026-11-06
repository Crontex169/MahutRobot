"""
PS4 Kolu (DualShock 4) ve Gamepad Donanım Test Betiği.

Bu betik:
1. Bilgisayara bağlı joystick / gamepad cihazlarını listeler.
2. PS4 kolundan gelen analog eksenleri, tetikleri ve buton basımlarını canlı olarak terminale yazdırır.
3. Çalıştırmak için:
   python Teleoperasyonlar/test_ps4.py
"""

import sys
import time
import os

os.environ["SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS"] = "1"

try:
    import pygame
except ImportError:
    print("[HATA] Pygame kütüphanesi yüklü değil!")
    print("Lütfen önce yükleyin: pip install pygame")
    sys.exit(1)


def main():
    pygame.init()
    pygame.joystick.init()

    joystick_count = pygame.joystick.get_count()
    print("=" * 60)
    print(f"🎮 Bulunan Gamepad Sayısı: {joystick_count}")
    print("=" * 60)

    if joystick_count == 0:
        print("\n[UYARI] Hiçbir gamepad algılanmadı!")
        print("Lütfen PS4 kolunuzu USB kablosuyla veya Bluetooth ile bağlayıp tekrar deneyin.")
        print("Linux'ta kontrol etmek için: ls -l /dev/input/js*")
        return

    for i in range(joystick_count):
        js = pygame.joystick.Joystick(i)
        js.init()
        print(f"[{i}] Cihaz Adı   : {js.get_name()}")
        print(f"    Eksen Sayısı: {js.get_numaxes()}")
        print(f"    Buton Sayısı: {js.get_numbuttons()}")
        print(f"    D-Pad Sayısı: {js.get_numhats()}")

    controller = pygame.joystick.Joystick(0)
    controller.init()

    print("\n[BİLGİ] 1. Cihaz dinleniyor. Çıkmak için Ctrl+C tuşlarına basın.\n")
    print("Denediğinizde değerlerin değiştiğini görmelisiniz:")
    print("-" * 60)

    try:
        while True:
            pygame.event.pump()

            # Eksenleri oku (Linux evdev / PS4 formatı)
            num_axes = controller.get_numaxes()
            lx = controller.get_axis(0) if num_axes > 0 else 0.0
            ly = controller.get_axis(1) if num_axes > 1 else 0.0
            l2 = controller.get_axis(2) if num_axes > 2 else 0.0
            rx = controller.get_axis(3) if num_axes > 3 else 0.0
            ry = controller.get_axis(4) if num_axes > 4 else 0.0
            r2 = controller.get_axis(5) if num_axes > 5 else 0.0

            # Basılı butonları bul
            pressed_buttons = [
                b for b in range(controller.get_numbuttons()) if controller.get_button(b)
            ]

            # D-Pad
            hat = controller.get_hat(0) if controller.get_numhats() > 0 else (0, 0)

            status = (
                f"\r[Sol Stick: X={lx:+0.2f}, Y={ly:+0.2f}] "
                f"[Sağ Stick: X={rx:+0.2f}, Y={ry:+0.2f}] "
                f"[L2={l2:+0.2f}, R2={r2:+0.2f}] "
                f"[D-Pad={hat}] "
                f"[Butonlar={pressed_buttons}]"
            )
            sys.stdout.write(status)
            sys.stdout.flush()

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\n[ÇIKIŞ] Test sonlandırıldı.")
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
