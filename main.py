import helper_functions as hpf
import pstats
import cProfile

if __name__ == '__main__':
    with cProfile.Profile() as pr:
        hpf.main(algorithm=1, shift_id=0, reversed_text=False, reverse_gematria=False, interrupter=0, ciphertext='hollow')
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats(filename='needs_profiling.prof')

# TODO: Adding Unittests to ensure that the encryptions are working as intended
# TODO: Adding Unittests to ensure that the decryptions are working as intended
# TODO: Add Ciphertext Autokey?
