import numpy as np
import numpy.typing as npt
import lp_text as lp
from prime_array import numpy_prime_array
import pandas as pd


def apply_shift(ct_numbers: npt.ArrayLike, shift_id: npt.ArrayLike, interrupter_array) -> np.ndarray:
    shift_index = 0

    prime = numpy_prime_array()

    if shift_id == 1:
        for index in range(ct_numbers.shape[1]):
            if interrupter_array[index]:
                continue
            ct_numbers[:, index] += prime[shift_index] - 1
            shift_index += 1
    elif shift_id == 2:
        for index in range(ct_numbers.shape[1]):
            if interrupter_array[index]:
                continue
            ct_numbers[:, index] -= prime[shift_index] - 1
            shift_index += 1
    elif shift_id == 3:
        for index in range(ct_numbers.shape[1]):
            if interrupter_array[index]:
                continue
            ct_numbers[:, index] += prime[shift_index]
            shift_index += 1
    elif shift_id == 4:
        for index in range(ct_numbers.shape[1]):
            if interrupter_array[index]:
                continue
            ct_numbers[:, index] -= prime[shift_index]
            shift_index += 1
    elif shift_id == 5:
        for index in range(ct_numbers.shape[1]):
            if interrupter_array[index]:
                continue
            ct_numbers[index] += shift_index
            shift_index += 1
    elif shift_id == 6:
        for index in range(ct_numbers.shape[1]):
            if interrupter_array[index]:
                continue
            ct_numbers[index] -= shift_index
            shift_index += 1
    return np.remainder(ct_numbers, 29)


class BestKeyStorage:
    def __init__(self):
        self.store = []
        self.N = 1000

    def add(self, item):
        self.store.append(item)
        self.store.sort(key=lambda x: x[0], reverse=True)
        self.store = self.store[:self.N]


def read_data_from_file(file_name: str) -> np.ndarray:
    with open(file_name) as f:
        lines = f.readlines()

    results = [line.split(',')[4] for line in lines]
    probabilities = np.zeros(len(results), dtype=float)
    for index, item in enumerate(results):
        probabilities[index] = float(item.replace('\n', ''))

    return probabilities


def decryption_autokey(keys: npt.ArrayLike, ct_numbers: npt.ArrayLike, current_interrupter: npt.ArrayLike) -> np.ndarray:
    mt = ct_numbers.copy()

    if len(mt.shape) == 1:
        mt = np.array([mt])
    if len(keys.shape) == 1:
        keys = np.array([keys])

    len_keys = keys.shape[1]
    indices = np.flatnonzero(np.logical_not(current_interrupter))

    for s, t in enumerate(indices[0:len_keys]):
        mt[:, t] = (mt[:, t] - keys[:, s]) % 29

    step_size = np.arange(len_keys, indices.shape[0], len_keys)

    if step_size[-1] != mt.shape[1]:
        step_size = np.append(step_size, indices.shape[0])

    diff_step_size = np.cumsum(np.concatenate(([0], np.diff(step_size))))

    for index in range(len(step_size) - 1):
        mt[:, indices[step_size[index]:step_size[index + 1]]] = \
            (mt[:, indices[step_size[index]:step_size[index + 1]]] - mt[:, diff_step_size[index]:diff_step_size[index + 1]]) % 29
    return mt


def decryption_vigenere(keys: npt.ArrayLike, ct_numbers: npt.ArrayLike, current_interrupter: npt.ArrayLike) -> np.ndarray:
    counter = 0
    key_shape = keys.shape
    key_length = key_shape[1]
    mt = np.tile(ct_numbers, (key_shape[0], 1))

    for index in range(len(ct_numbers)):
        if current_interrupter[index]:
            continue
        else:
            mt[:, index] = (mt[:, index] - keys[:, counter % key_length]) % 29
            counter += 1
    return mt


def calculate_fitness(childkey: npt.ArrayLike, ct_numbers: npt.ArrayLike, probabilities: npt.ArrayLike, algorithm: int,
                      current_interrupter: npt.ArrayLike, reversed_text: bool, shift_id: int) -> np.ndarray:
    if algorithm == 0:
        mt = decryption_vigenere(childkey, ct_numbers, current_interrupter)
    elif algorithm == 1:
        mt = decryption_autokey(childkey, ct_numbers, current_interrupter)
    else:
        raise AssertionError()

    if shift_id > 0:
        mt = apply_shift(mt, shift_id, current_interrupter)
    if reversed_text:
        mt = mt[:, ::-1]
    len_ciphertext = mt.shape[1]
    indices = np.array(
        [mt[:, 0:len_ciphertext - 3] * 24389, mt[:, 1:len_ciphertext - 2] * 841, mt[:, 2:len_ciphertext - 1] * 29, mt[:, 3:len_ciphertext]])
    score = np.sum(probabilities[np.sum(indices, axis=0)], axis=1)
    return score


def translate_to_english(parent_key: npt.ArrayLike, reverse_gematria: bool) -> str:
    dic = ["F", "U", "TH", "O", "R", "C", "G", "W", "H", "N", "I", "J", "EO", "P", "X", "S", "T", "B", "E", "M", "L",
           "ING", "OE", "D", "A", "AE", "Y", "IA", "EA"]
    if reverse_gematria:
        dic.reverse()
    translation = ""
    for index in np.nditer(parent_key):
        translation += dic[index]
    return translation


def translate_best_text(algorithm: int, best_key_ever: npt.ArrayLike, ct_numbers: npt.ArrayLike, current_interrupter: npt.ArrayLike,
                        reverse_gematria: bool) -> str:
    if algorithm == 0:
        return translate_to_english(decryption_vigenere(best_key_ever, ct_numbers, current_interrupter), reverse_gematria)
    if algorithm == 1:
        return translate_to_english(decryption_autokey(best_key_ever, ct_numbers, current_interrupter), reverse_gematria)
    else:
        print('Invalid algorithm ID')


def finding_keys(counting: int, ct_numbers: npt.ArrayLike, ct_interrupters: npt.ArrayLike, number_of_interrupters: int,
                 probabilities: npt.ArrayLike, algorithm: int, reversed_text: bool, reverse_gematria: bool, shift_id: int):
    current_interrupter = np.copy(ct_interrupters)
    bit_rep = bin(int(counting))[2:].zfill(number_of_interrupters)
    current_interrupter[ct_interrupters == 1] = np.array(list(map(int, bit_rep)))
    best_score_ever = -1000000.0
    best_key_ever = np.empty(1)

    for key_length in range(1, 20):

        parent_key = np.random.randint(28, size=(1, key_length))
        parent_score = calculate_fitness(parent_key, ct_numbers, probabilities, algorithm, current_interrupter, reversed_text, shift_id)

        still_improving = True
        ct_numbers_large = np.tile(ct_numbers, (29, 1))

        while still_improving:
            for index in range(key_length):
                childkey = np.tile(parent_key, (29, 1))

                childkey[:, index] = np.arange(29)
                scores = calculate_fitness(childkey, ct_numbers_large, probabilities, algorithm, current_interrupter, reversed_text, shift_id)

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
                best_text = translate_best_text(algorithm, best_key_ever, ct_numbers, current_interrupter, reverse_gematria)
                key_translated = translate_to_english(best_key_ever, reverse_gematria)
                best_keys.add((best_score_ever, best_key_ever.shape[1], best_key_ever, best_text, key_translated))
    print(counting)


def main(algorithm, shift_id, reversed_text, reverse_gematria, interrupter, ciphertext):
    global best_keys
    best_keys = BestKeyStorage()

    if isinstance(ciphertext, str):
        ct_numbers = get_ciphertext(ciphertext)
    else:
        ct_numbers = ciphertext

    ct_interrupters = (ct_numbers == interrupter)
    number_of_interrupters = sum(ct_interrupters)
    print(f"Number of interrupters: {number_of_interrupters}")
    if reversed_text:
        ct_numbers = reverse_ct(ct_numbers)

    probabilities = read_data_from_file("new_quadgrams.txt")
    if reverse_gematria:
        probabilities = probabilities[::-1]

    for counting in range(pow(2, number_of_interrupters)):
        finding_keys(counting, ct_numbers, ct_interrupters, number_of_interrupters, probabilities, algorithm, reversed_text, reverse_gematria,
                     shift_id)

    keys_rating = []
    keys_length = []
    keys_mt = []
    keys_key = []
    keys_key_translate = []

    for index in range(len(best_keys.store)):
        keys_rating.append(best_keys.store[index][0])
        keys_length.append(best_keys.store[index][1])
        keys_key.append(best_keys.store[index][2])
        keys_mt.append(best_keys.store[index][3])
        keys_key_translate.append(best_keys.store[index][4])

    df = pd.DataFrame({'Rating': keys_rating, 'Key length': keys_length, 'Message text': keys_mt, 'Keys in indices': keys_key,
                       'Keys translated': keys_key_translate})
    df.to_csv('keys.txt', index=False)


def reverse_ct(ct_numbers):
    return ct_numbers[::-1]


def get_ciphertext(textname):
    if textname.lower() == 'divinity':
        return lp.get_divinity_text()
    elif textname.lower() == 'koan2':
        return lp.get_koan2_text()
    elif textname.lower() == 'cross':
        return lp.get_cross_text()
    elif textname.lower() == 'spirals':
        return lp.get_spirals_text()
    elif textname.lower() == 'branches':
        return lp.get_branches_text()
    elif textname.lower() == 'mobius':
        return lp.get_mobius_text()
    elif textname.lower() == 'mayfly':
        return lp.get_mayfly_text()
    elif textname.lower() == 'wing_tree' or textname.lower() == 'wingtree':
        return lp.get_wing_tree_text()
    elif textname.lower() == 'cuneiform':
        return lp.get_cuneiform_text()
    elif textname.lower() == 'spiral_branches' or textname.lower() == 'spiralbranches':
        return lp.get_spiral_branches_text()
    elif textname.lower() == 'hollow':
        return lp.get_hollow_text()
    elif textname.lower() == 'an end':
        return lp.get_an_end_text()
    elif textname.lower() == 'parable':
        return lp.get_parable_text()
    else:
        print('Error')


def collect_results(result):
    best_keys.add(result)
