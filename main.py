import pstats
import numpy as np
import helper_functions as hpf
import lp_text
import timeit
import cProfile


def main():
    start = timeit.default_timer()
    algorithm = 1  # Vigenere, Autokey, Autokey Ciphertext
    shift_id = 0  # 0 Without #1 +Totient shift, #2 -Totient shift, #3 +prime shift, #4 -prime shift, #5 +index shift, #6 -index shift
    reversed_text = False
    reverse_gematria = False
    interrupter = 0

    ct_numbers = lp_text.get_hollow_text()
    ct_interrupters = np.int8((ct_numbers == interrupter))
    number_of_interrupters = sum(ct_interrupters)

    if reversed_text:
        ct_numbers = ct_numbers[::-1]

    if shift_id > 0:
        ct_numbers = hpf.apply_shift(ct_numbers, shift_id)

    probabilities = hpf.read_data_from_file("new_quadgrams.txt")
    if reverse_gematria:
        probabilities = probabilities[::-1]

    best_keys = hpf.BestKeyStorage()

    for counting in range(pow(2, number_of_interrupters)):
        current_interrupter = np.copy(ct_interrupters)
        bit_rep = bin(int(counting))[2:].zfill(number_of_interrupters)
        current_interrupter[ct_interrupters == 1] = np.array(list(bit_rep))
        best_score_ever = -1000000.0
        best_key_ever = []
        for key_length in range(2, 20):
            parent_key = np.random.randint(28, size=(1, key_length))  # np.array([[0, 10, 4, 0, 1, 19, 0, 18, 4, 18, 9, 0, 18]])
            parent_score = hpf.calculate_fitness(parent_key, ct_numbers, probabilities, algorithm,
                                                 current_interrupter, reversed_text)

            still_improving = True

            while still_improving:
                for index in range(key_length):
                    childkey = np.zeros((29, key_length), dtype=int)
                    childkey[:] = parent_key

                    childkey[:, index] = np.arange(29)
                    scores = hpf.calculate_fitness(childkey, ct_numbers, probabilities, algorithm, current_interrupter, reversed_text)

                    best_children_score = np.max(scores)

                    if best_children_score > parent_score:
                        k = np.where(scores == best_children_score)
                        parent_key = childkey[k[0], :]
                        parent_score = best_children_score

                if parent_score > best_score_ever:
                    best_score_ever = parent_score
                    best_key_ever = parent_key
                else:
                    still_improving = False
                    translation = hpf.translate_best_text(algorithm, best_key_ever, ct_numbers, current_interrupter, reverse_gematria)
                    best_keys.add((best_score_ever, best_key_ever.shape[1], best_key_ever, translation))
        print(counting)

    f = open('keys.txt', 'w')
    for t in best_keys.store:
        f.write(' '.join(str(s) for s in t) + '\n')
    f.close()

    stop = timeit.default_timer()

    print('Time: ', stop - start)


if __name__ == '__main__':
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()
