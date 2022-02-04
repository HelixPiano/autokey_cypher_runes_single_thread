from unittest import TestCase
import numpy as np
import encryption as enc
import testing_texts as tt


class TestEncryption(TestCase):

    def test_vigenere_encryption(self):
        interrupter = 0
        plaintext = tt.get_koan2_plaintext()
        ct = plaintext.copy()
        key = np.array([0, 10, 4, 0, 1, 19, 0, 18, 4, 18, 9, 0, 18])  # FIRFUMFERENFE
        ct = enc.vigenere_encryption(ct, interrupter, key)
        np.testing.assert_array_equal(tt.get_koan2_text(), ct)

    def test_autokey_encryption_beginning(self):
        interrupter = 0
        plaintext = tt.get_koan2_plaintext()
        ct = plaintext.copy()
        key = np.array([0, 10, 4, 0, 1, 19, 0, 18, 4, 18, 9, 0, 18])  # FIRFUMFERENFE
        ct_autokey = enc.autokey_encryption(ct, interrupter, key)
        ct_vigenere = enc.vigenere_encryption(ct, interrupter, key)
        np.testing.assert_array_equal(ct_autokey[0:len(key)], ct_vigenere[0:len(key)])

    def test_autokey_encryption_without_interrupters(self):
        interrupter = None
        plaintext = tt.test_autokey_encryption_plaintext()
        ct = plaintext.copy()
        key = np.array([18, 9, 20, 10, 6, 8, 16, 18, 9, 19, 18, 9, 16, 24, 7, 24, 10, 16, 15])
        ct_autokey = enc.autokey_encryption(ct, interrupter, key)
        np.testing.assert_array_equal(ct_autokey, tt.test_autokey_encryption_ct())

    def test_autokey_encryption_with_interrupters(self):
        interrupter = 0
        plaintext = tt.test_autokey_encryption_plaintext()
        ct = plaintext.copy()
        key = np.array([18, 9, 20, 10, 6, 8, 16, 18, 9, 19, 18, 9, 16, 24, 7, 24, 10, 16, 15])
        ct_autokey = enc.autokey_encryption(ct, interrupter, key)
        np.testing.assert_array_equal(ct_autokey, tt.test_autokey_encryption_ct())

    def test_translate_to_english(self):
        self.assertEqual(enc.translate_to_english(np.array([0, 10, 4, 0, 1, 19, 0, 18, 4, 18, 9, 0, 18]), reverse_gematria=False).upper(),
                         'FIRFUMFERENFE')

    def test_atbash(self):
        np.testing.assert_array_equal(enc.atbash(tt.get_a_warning_plaintext()), tt.get_a_warning_atbash())

    def test_convert_text_to_index(self):
        np.testing.assert_array_equal(enc.convert_text_to_index('FIRFUMFERENFE'), np.array([0, 10, 4, 0, 1, 19, 0, 18, 4, 18, 9, 0, 18]))

    def test_full_encrypt_text(self):
        interrupter = 0
        plaintext = tt.get_koan2_plaintext()
        key = np.array([0, 10, 4, 0, 1, 19, 0, 18, 4, 18, 9, 0, 18])  # FIRFUMFERENFE
        algorithm = 'Vigenere'
        reverse_text = False
        reverse_gematria = False
        shift_id = 0
        ct = enc.encrypt_text(plaintext, algorithm, shift_id, reverse_text, reverse_gematria, interrupter, key)
        np.testing.assert_array_equal(tt.get_koan2_text(), ct)

    def test_shift_add_totient(self):
        interrupter = 0
        ct = np.array([1, 2, 3, 4, 5, 0, 6, 7, 8, 9])
        ct = enc.apply_shift(ct, interrupter, shift_id=1)
        np.testing.assert_array_equal(ct, np.array([2, 4, 7, 10, 15, 0, 18, 23, 26, 2]))

    def test_shift_subtract_totient(self):
        interrupter = 0
        ct = np.array([1, 2, 3, 4, 5, 0, 6, 7, 8, 9])
        ct = enc.apply_shift(ct, interrupter, shift_id=2)
        np.testing.assert_array_equal(ct, np.array([0, 0, 28, 27, 24, 0, 23, 20, 19, 16]))

    def test_shift_add_prime(self):
        interrupter = 0
        ct = np.array([1, 2, 3, 4, 5, 0, 6, 7, 8, 9])
        ct = enc.apply_shift(ct, interrupter, shift_id=3)
        np.testing.assert_array_equal(ct, np.array([3, 5, 8, 11, 16, 0, 19, 24, 27, 3]))

    def test_shift_subtract_prime(self):
        interrupter = 0
        ct = np.array([1, 2, 3, 4, 5, 0, 6, 7, 8, 9])
        ct = enc.apply_shift(ct, interrupter, shift_id=4)
        np.testing.assert_array_equal(ct, np.array([28, 28, 27, 26, 23, 0, 22, 19, 18, 15]))

    def test_shift_add_index(self):
        interrupter = 0
        ct = np.array([1, 2, 3, 4, 5, 0, 6, 7, 8, 9])
        ct = enc.apply_shift(ct, interrupter, shift_id=5)
        np.testing.assert_array_equal(ct, np.array([1, 3, 5, 7, 9, 0, 11, 13, 15, 17]))

    def test_shift_subtract_index(self):
        interrupter = 0
        ct = np.array([1, 2, 3, 4, 5, 0, 6, 7, 8, 9])
        ct = enc.apply_shift(ct, interrupter, shift_id=6)
        np.testing.assert_array_equal(ct, np.array([1, 1, 1, 1, 1, 0, 1, 1, 1, 1]))

# TODO Add encryption that tests combination of shift+autokey/vigenere
# TODO Test Autokey encryption that uses interrupters