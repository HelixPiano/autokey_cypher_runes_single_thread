import helper_functions as hpf
import pstats
import cProfile
import numpy as np
import testing_texts as tt

if __name__ == '__main__':
    # with cProfile.Profile() as pr:
    #     hpf.main(algorithm=1, shift_id=0, reversed_text=False, reverse_gematria=False, interrupter=0, ciphertext='hollow')
    interrupter = None
    key = np.array([[18, 9, 20, 10, 6, 8, 16, 18, 9, 19, 18, 9, 16, 24, 7, 24, 10, 16, 15]])
    plaintext = tt.test_autokey_encryption_plaintext()
    ct = tt.test_autokey_encryption_ct()
    interrupter_array = np.asarray(plaintext == interrupter, dtype=int)
    with cProfile.Profile() as pr:
        for i in range(20000):
            mt = hpf.decryption_autokey(key, ct, interrupter_array)
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats(filename='needs_profiling.prof')

# TODO: Adding Unittests to ensure that the encryptions are working as intended
# TODO: Adding Unittests to ensure that the decryptions are working as intended
# TODO: Run everything again for Hollow
# TODO: Add Ciphertext Autokey?
