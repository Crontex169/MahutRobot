"""
Birim testleri: Türkçe Komut Çözücü ve Normalizasyon Modülü (CommandParser).
"""

import unittest
from mahut_kol.speech.command_parser import CommandParser, turkish_lower


class TestCommandParser(unittest.TestCase):
    def setUp(self):
        self.parser = CommandParser()

    def test_turkish_lower(self):
        """Türkçe karakterlerin (özellikle İ -> i, I -> ı) doğru dönüştürüldüğünü test et."""
        self.assertEqual(turkish_lower("İSTANBUL IŞIK"), "istanbul ışık")
        self.assertEqual(turkish_lower("KIRMIZI KÜP"), "kırmızı küp")
        self.assertEqual(turkish_lower("YEŞİL SİLİNDİR"), "yeşil silindir")
        self.assertEqual(turkish_lower("MAVİ PRİZMA"), "mavi prizma")

    def test_pick_and_place_command(self):
        """Kırmızı küpü al yeşil silindirin sağına koy cümlesini doğrula."""
        cmd = self.parser.parse("Kırmızı küpü al yeşil silindirin sağına koy")
        self.assertTrue(cmd["valid"])
        self.assertEqual(cmd["action"], "pick_and_place")
        self.assertEqual(cmd["manipulated_object"], "red_cube")
        self.assertEqual(cmd["reference_object"], "green_cylinder")
        self.assertEqual(cmd["relation"], "RIGHT")
        self.assertEqual(cmd["target_slot"], "green_cylinder_RIGHT")

    def test_push_command(self):
        """Mavi silindiri it komutunu doğrula."""
        cmd = self.parser.parse("Mavi silindiri it")
        self.assertTrue(cmd["valid"])
        self.assertEqual(cmd["action"], "push")
        self.assertEqual(cmd["manipulated_object"], "blue_cylinder")
        self.assertIsNone(cmd["reference_object"])
        self.assertIsNone(cmd["relation"])

    def test_pick_command(self):
        """Yeşil prizmayı kaldır komutunu doğrula."""
        cmd = self.parser.parse("Yeşil prizmayı kaldır!")
        self.assertTrue(cmd["valid"])
        self.assertEqual(cmd["action"], "pick")
        self.assertEqual(cmd["manipulated_object"], "green_triangular_prism")

    def test_all_nine_objects(self):
        """Tüm 9 nesne kombinasyonunu (3 renk x 3 şekil) test et."""
        colors = [("kırmızı", "red"), ("yeşil", "green"), ("mavi", "blue")]
        shapes = [
            ("küp", "cube"),
            ("silindir", "cylinder"),
            ("üçgen prizma", "triangular_prism"),
        ]

        for tr_col, en_col in colors:
            for tr_shp, en_shp in shapes:
                phrase = f"{tr_col} {tr_shp}ü tut" if "küp" in tr_shp else f"{tr_col} {tr_shp} kaldır"
                res = self.parser.parse(phrase)
                expected_obj = f"{en_col}_{en_shp}"
                self.assertTrue(
                    res["valid"],
                    f"Başarısız ifade: {phrase}, çıktı: {res}"
                )
                self.assertEqual(res["manipulated_object"], expected_obj)

    def test_all_five_relations(self):
        """Tüm 5 bağıl yönü doğrula: FRONT, BACK, LEFT, RIGHT, TOP."""
        directions = [
            ("önüne", "FRONT"),
            ("arkasına", "BACK"),
            ("soluna", "LEFT"),
            ("sağına", "RIGHT"),
            ("üstüne", "TOP"),
            ("üzerine", "TOP"),
        ]

        for tr_dir, en_dir in directions:
            cmd = self.parser.parse(f"Mavi küpü al kırmızı silindirin {tr_dir} bırak")
            self.assertTrue(cmd["valid"])
            self.assertEqual(cmd["action"], "pick_and_place")
            self.assertEqual(cmd["manipulated_object"], "blue_cube")
            self.assertEqual(cmd["reference_object"], "red_cylinder")
            self.assertEqual(cmd["relation"], en_dir)

    def test_suffix_variations(self):
        """Farklı Türkçe çekim eklerinin tanınmasını test et."""
        variations = [
            ("kırmızı küpü al", "red_cube"),
            ("kırmızı küp al", "red_cube"),
            ("kırmızı küpe dokun", "red_cube"),
            ("kırmızı küpten", "red_cube"),
            ("yeşil silindiri", "green_cylinder"),
            ("yeşil silindirin", "green_cylinder"),
            ("mavi prizmayı", "blue_triangular_prism"),
            ("mavi üçgeni", "blue_triangular_prism"),
        ]
        for phrase, expected_obj in variations:
            extracted = self.parser.extract_objects(phrase)
            self.assertTrue(len(extracted) >= 1, f"Nesne bulunamadı: {phrase}")
            self.assertEqual(extracted[0]["object_id"], expected_obj)

    def test_invalid_and_incomplete_commands(self):
        """Eksik veya geçersiz komutlarda valid=False ve hata mesajı dönmeli."""
        # Yalnızca nesne var, eylem yok
        res = self.parser.parse("Kırmızı küp")
        self.assertFalse(res["valid"])
        self.assertIn("error", res)

        # Anlamsız metin
        res = self.parser.parse("Merhaba nasılsın bugün hava güzel")
        self.assertFalse(res["valid"])
        self.assertIn("error", res)

        # Hedef yön eksik
        res = self.parser.parse("Kırmızı küpü yeşil silindirin yanına koy")
        # 'yanına' net bir bağıl yön değil, net relation belirtilmeli
        self.assertFalse(res["valid"])

        # Referans nesne olmadan sadece yön verilen eksik pick_and_place
        res2 = self.parser.parse("Kırmızı küpü sağına koy")
        self.assertFalse(res2["valid"])
        self.assertIn("error", res2)

    def test_relative_locative_suffixes(self):
        """-ndaki, -ndeki, -undaki vb. aitlik/yön eklerini doğrula."""
        locative_cases = [
            ("Mavi küpü al kırmızı silindirin sağındaki slota koy", "RIGHT"),
            ("Mavi küpü al kırmızı silindirin solundaki alana bırak", "LEFT"),
            ("Mavi küpü al kırmızı silindirin önündeki yere koy", "FRONT"),
            ("Mavi küpü al kırmızı silindirin arkasındaki alana bırak", "BACK"),
            ("Mavi küpü al kırmızı silindirin üstündeki tepeye koy", "TOP"),
        ]
        for phrase, expected_rel in locative_cases:
            cmd = self.parser.parse(phrase)
            self.assertTrue(cmd["valid"], f"Başarısız ifade: {phrase}, çıktı: {cmd}")
            self.assertEqual(cmd["relation"], expected_rel)


if __name__ == "__main__":
    unittest.main()
