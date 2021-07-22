import pstats
import numpy as np
import helper_functions as hpf
import lp_text
import timeit
import cProfile


def main():
    start = timeit.default_timer()
    algorithm = 0  # Vigenere, Autokey, Autokey Ciphertext
    shift_id = 0  # 0 Without #1 +Totient shift, #2 -Totient shift, #3 +prime shift, #4 -prime shift, #5 +index shift, #6 -index shift
    reversed_text = False
    reverse_gematria = False
    interrupter = 0

    CT_numbers = lp_text.get_koan2_text()
    CT_interrupters = np.int8((CT_numbers == interrupter))
    number_of_interrupters = sum(CT_interrupters)
    interrupters = np.zeros((pow(2, number_of_interrupters), len(CT_numbers)), dtype=np.uint8)

    if reversed_text:
        CT_numbers = CT_numbers[::-1]

    if shift_id > 0:
        CT_numbers = hpf.apply_shift(CT_numbers, shift_id)

    probabilities = hpf.read_data_from_file("new_quadgrams.txt")
    if reverse_gematria:
        probabilities = probabilities[::-1]

    for index in range(pow(2, number_of_interrupters)):
        my_dude = np.copy(CT_interrupters)
        bit_rep = bin(int(index))[2:].zfill(number_of_interrupters)
        my_dude[my_dude == 1] = np.array(list(bit_rep))
        interrupters[index] = my_dude

    best_keys = hpf.best_key_storage()

    for counting in range(pow(2, number_of_interrupters)):
        current_interrupter = interrupters[counting]
        best_score_ever = -1000000.0
        best_key_ever = []
        for key_length in range(2, 20):
            parent_key = np.random.randint(28, size=(1, key_length))
            parent_score = hpf.calculate_fitness(parent_key, CT_numbers, probabilities, algorithm,
                                                 current_interrupter, reversed_text)

            still_improving = True

            while still_improving:
                for index in range(len(parent_key)):
                    childkey = np.zeros((29, len(parent_key[0])), dtype=int)
                    childkey[:] = parent_key

                    childkey[:, index] = np.arange(29)
                    scores = hpf.calculate_fitness(childkey, CT_numbers, probabilities, algorithm, current_interrupter, reversed_text)

                    best_children_score = np.max(scores)

                    if best_children_score > parent_score:
                        k = np.where(scores == best_children_score)
                        parent_key = childkey[k[0][0], :]
                        parent_score = best_children_score

                if parent_score > best_score_ever:
                    best_score_ever = parent_score
                    best_key_ever = parent_key
                else:
                    still_improving = False
                    Translation = hpf.translate_best_text(algorithm, best_key_ever, CT_numbers, current_interrupter, reverse_gematria)
                    best_keys.add((best_score_ever, len(best_key_ever), best_key_ever, Translation))
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
