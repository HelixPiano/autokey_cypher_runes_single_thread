from unittest import TestCase
import numpy as np
import encryption as enc
import testing_texts as tt
import helper_functions as hpf
import pandas as pd


class TestDecryption(TestCase):
    def test_pure_vigenere_decryption(self):
        interrupter = 0
        key = np.array([[0, 10, 4, 0, 1, 19, 0, 18, 4, 18, 9, 0, 18]])
        plaintext = tt.get_koan2_plaintext()
        ct = tt.get_koan2_text()
        interrupter_array = (plaintext == interrupter).astype(int)
        mt = hpf.decryption_vigenere(key, ct, interrupter_array)
        np.testing.assert_array_equal(plaintext, np.squeeze(mt))

    def test_vigenere_decryption_koan2(self):
        interrupter = 0
        plaintext = tt.get_koan2_plaintext()
        ct = tt.get_koan2_text()
        hpf.main(algorithm=0, shift_id=0, reversed_text=False, reverse_gematria=False, interrupter=interrupter,
                 ciphertext=ct)
        df = pd.read_csv('keys.txt', sep=',')
        self.assertEqual(hpf.translate_to_english(tt.get_koan2_plaintext(), reverse_gematria=False), df.loc[0, 'Message text'])

    def test_pure_standard_autokey(self):
        interrupter = None
        key = np.array([[18, 9, 20, 10, 6, 8, 16, 18, 9, 19, 18, 9, 16, 24, 7, 24, 10, 16, 15]])
        plaintext = tt.test_autokey_encryption_plaintext()
        ct = tt.test_autokey_encryption_ct()
        interrupter_array = np.asarray(plaintext == interrupter, dtype=int)
        mt = hpf.decryption_autokey(key, ct, interrupter_array)
        np.testing.assert_array_equal(plaintext, np.squeeze(mt))

    def test_autokey(self):
        hpf.main(algorithm=1, shift_id=0, reversed_text=False, reverse_gematria=False, interrupter=None,
                 ciphertext=tt.test_autokey_encryption_ct())
        df = pd.read_csv('keys.txt', sep=',')
        self.assertEqual(hpf.translate_to_english(tt.test_autokey_encryption_plaintext(), reverse_gematria=False), df.loc[0, 'Message text'])

    def test_autokey_interrupter(self):
        key = np.array([20, 10, 17, 18, 4])  # Liber
        interrupter = 0
        ct = enc.autokey_encryption(tt.get_koan2_plaintext(), interrupter, key)
        interrupter_array = np.asarray(tt.get_koan2_plaintext() == interrupter, dtype=bool)
        mt = hpf.decryption_autokey(np.array([key]), np.array([ct]), interrupter_array)
        mt = mt.squeeze()
        np.testing.assert_array_equal(tt.get_koan2_plaintext(), mt)

    def test_autokey_full(self):
        key = np.array([20, 10, 17, 18, 4])  # Liber
        interrupter = 0
        ct = enc.autokey_encryption(tt.get_koan2_plaintext(), interrupter, key)

        hpf.main(algorithm=1, shift_id=0, reversed_text=False, reverse_gematria=False, interrupter=0,
                 ciphertext=ct)
        df = pd.read_csv('keys.txt', sep=',')
        self.assertEqual(hpf.translate_to_english(tt.get_koan2_plaintext(), reverse_gematria=False), df.loc[0, 'Message text'])

    def test_autokey_shift0(self):
        interrupter = 0
        ct_numbers = tt.get_koan2_plaintext()
        interrupter_array = np.asarray(ct_numbers == interrupter)
        ciphertext = enc.encrypt_text(plaintext=ct_numbers, algorithm='Autokey', shift_id=0, reverse_text=False, reverse_gematria=False,
                                      interrupter=interrupter, key=np.array([0, 10, 15, 8, 18, 4, 19, 24, 9]),
                                      interrupter_array=interrupter_array)
        hpf.main(algorithm=1, shift_id=0, reversed_text=False, reverse_gematria=False, interrupter=interrupter, ciphertext=ciphertext)
        df = pd.read_csv('keys.txt', sep=',')
        self.assertEqual(hpf.translate_to_english(tt.get_koan2_plaintext(), reverse_gematria=False), df.loc[0, 'Message text'])

    def test_autokey_shift1(self):
        interrupter = 28
        ct_numbers = tt.get_koan2_plaintext()
        interrupter_array = np.asarray(ct_numbers == interrupter)
        ciphertext = enc.encrypt_text(plaintext=ct_numbers, algorithm='Autokey', shift_id=1, reverse_text=False, reverse_gematria=False,
                                      interrupter=interrupter, key=np.array([0, 10, 15, 8, 18, 4, 19, 24, 9]),
                                      interrupter_array=interrupter_array)
        hpf.main(algorithm=1, shift_id=1, reversed_text=False, reverse_gematria=False, interrupter=interrupter, ciphertext=ciphertext)
        df = pd.read_csv('keys.txt', sep=',')
        self.assertEqual(hpf.translate_to_english(tt.get_koan2_plaintext(), reverse_gematria=False), df.loc[0, 'Message text'])

    def test_autokey_shift2(self):
        interrupter = 12
        ct_numbers = tt.get_koan2_plaintext()
        interrupter_array = np.asarray(ct_numbers == interrupter)
        ciphertext = enc.encrypt_text(plaintext=ct_numbers, algorithm='Autokey', shift_id=2, reverse_text=False, reverse_gematria=False,
                                      interrupter=interrupter, key=np.array([0, 10, 15, 8, 18, 4, 19, 24, 9]),
                                      interrupter_array=interrupter_array)
        hpf.main(algorithm=1, shift_id=2, reversed_text=False, reverse_gematria=False, interrupter=interrupter, ciphertext=ciphertext)
        df = pd.read_csv('keys.txt', sep=',')
        self.assertEqual(hpf.translate_to_english(tt.get_koan2_plaintext(), reverse_gematria=False), df.loc[0, 'Message text'])

    def test_autokey_shift3(self):
        interrupter = 27
        ct_numbers = tt.get_koan2_plaintext()
        interrupter_array = np.asarray(ct_numbers == interrupter)
        ciphertext = enc.encrypt_text(plaintext=ct_numbers, algorithm='Autokey', shift_id=3, reverse_text=False, reverse_gematria=False,
                                      interrupter=interrupter, key=np.array([0, 10, 15, 8, 18, 4, 19, 24, 9]),
                                      interrupter_array=interrupter_array)
        hpf.main(algorithm=1, shift_id=3, reversed_text=False, reverse_gematria=False, interrupter=interrupter, ciphertext=ciphertext)
        df = pd.read_csv('keys.txt', sep=',')
        self.assertEqual(hpf.translate_to_english(tt.get_koan2_plaintext(), reverse_gematria=False), df.loc[0, 'Message text'])

    def test_autokey_shift4(self):
        interrupter = 11
        ct_numbers = tt.get_koan2_plaintext()
        interrupter_array = np.asarray(ct_numbers == interrupter)
        ciphertext = enc.encrypt_text(plaintext=ct_numbers, algorithm='Autokey', shift_id=4, reverse_text=False, reverse_gematria=False,
                                      interrupter=interrupter, key=np.array([0, 10, 15, 8, 18, 4, 19, 24, 9]),
                                      interrupter_array=interrupter_array)
        hpf.main(algorithm=1, shift_id=4, reversed_text=False, reverse_gematria=False, interrupter=interrupter, ciphertext=ciphertext)
        df = pd.read_csv('keys.txt', sep=',')
        self.assertEqual(hpf.translate_to_english(tt.get_koan2_plaintext(), reverse_gematria=False), df.loc[0, 'Message text'])

    def test_autokey_shift5(self):
        interrupter = 25
        ct_numbers = tt.get_koan2_plaintext()
        interrupter_array = np.asarray(ct_numbers == interrupter)
        ciphertext = enc.encrypt_text(plaintext=ct_numbers, algorithm='Autokey', shift_id=5, reverse_text=False, reverse_gematria=False,
                                      interrupter=interrupter, key=np.array([0, 10, 15, 8, 18, 4, 19, 24, 9]),
                                      interrupter_array=interrupter_array)
        hpf.main(algorithm=1, shift_id=5, reversed_text=False, reverse_gematria=False, interrupter=interrupter, ciphertext=ciphertext)
        df = pd.read_csv('keys.txt', sep=',')
        self.assertEqual(hpf.translate_to_english(tt.get_koan2_plaintext(), reverse_gematria=False), df.loc[0, 'Message text'])

    def test_autokey_shift6(self):
        interrupter = 22
        ct_numbers = tt.get_koan2_plaintext()
        interrupter_array = np.asarray(ct_numbers == interrupter)
        ciphertext = enc.encrypt_text(plaintext=ct_numbers, algorithm='Autokey', shift_id=6, reverse_text=False, reverse_gematria=False,
                                      interrupter=interrupter, key=np.array([0, 10, 15, 8, 18, 4, 19, 24, 9]),
                                      interrupter_array=interrupter_array)
        hpf.main(algorithm=1, shift_id=6, reversed_text=False, reverse_gematria=False, interrupter=interrupter, ciphertext=ciphertext)
        df = pd.read_csv('keys.txt', sep=',')
        self.assertEqual(hpf.translate_to_english(tt.get_koan2_plaintext(), reverse_gematria=False), df.loc[0, 'Message text'])
