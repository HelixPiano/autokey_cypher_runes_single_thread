import numpy as np
from sympy.ntheory.factor_ import totient
from sympy import prime


def apply_shift(ct_numbers, shift_id):
    if shift_id == 1:
        for index in range(len(ct_numbers)):
            ct_numbers[index] += totient(prime(index + 1))
    if shift_id == 2:
        for index in range(len(ct_numbers)):
            ct_numbers[index] -= totient(prime(index + 1))
    if shift_id == 3:
        for index in range(len(ct_numbers)):
            ct_numbers[index] += prime(index + 1)
    if shift_id == 4:
        for index in range(len(ct_numbers)):
            ct_numbers[index] -= prime(index + 1)
    if shift_id == 5:
        for index in range(len(ct_numbers)):
            ct_numbers[index] += index
    if shift_id == 6:
        for index in range(len(ct_numbers)):
            ct_numbers[index] -= index
    return np.remainder(ct_numbers, 29)


class best_key_storage:
    def __init__(self):
        self.store = []
        self.N = 1000

    def add(self, item):
        self.store.append(item)
        self.store.sort(key=lambda x: x[0], reverse=True)
        self.store = self.store[:self.N]


def read_data_from_file(file_name):
    s = open(file_name, "r")
    lines = s.readlines()

    # ints = np.asarray(([line.split(',')[0:4] for line in lines]), dtype=int, order='C')
    probabilities = [line.split(',')[4] for line in lines]
    for index in range(len(probabilities)):
        probabilities[index] = float(probabilities[index].replace('\n', ''))
    probabilities = np.array(probabilities)
    s.close()
    return probabilities


def decryption_autokey(key, ct_numbers, current_interrupter):
    counter = 0
    index = 0
    key_shape = key.shape
    if len(key_shape) == 1:
        key_length = key_shape[0]
        mt = np.zeros((1, len(ct_numbers)), dtype=int)
    else:
        key_length = key_shape[1]
        mt = np.zeros((29, len(ct_numbers)), dtype=int)
    mt[:] = ct_numbers

    if np.sum(current_interrupter[0:len(key)]) == 0:
        mt[:, 0:key_length] = (mt[:, 0:key_length] - key) % 29
        index = key_length
    else:
        while counter < key.shape[1]:
            if current_interrupter[index] == 1:
                index += 1
                continue
            mt[:, index] = (mt[:, index] - key[counter]) % 29
            index += 1
            counter += 1

    position = 0

    for i in range(index, len(ct_numbers)):
        if current_interrupter[i] == 1:
            continue
        mt[:, i] = (mt[:, i] - mt[:, position]) % 29
        position += 1

    return mt


def decryption_vigenere(key, ct_numbers, current_interrupter):
    counter = 0
    key_shape = key.shape
    if len(key_shape) == 1:
        key_length = key_shape[0]
        mt = np.zeros((1, len(ct_numbers)), dtype=int)
        keys = np.zeros((1, key_shape[0]), dtype=int)
    else:
        key_length = key_shape[1]
        mt = np.zeros((29, len(ct_numbers)), dtype=int)
        keys = np.zeros((29, key_shape[1]), dtype=int)
    keys[:] = key
    mt[:] = ct_numbers

    for index in range(len(ct_numbers)):
        if current_interrupter[index]:
            continue
        else:
            mt[:, index] = (mt[:, index] - keys[:, counter % key_length]) % 29
            counter += 1
    return mt


def decryption_autokey_ciphertext(childkey, ct_numbers):
    key_shape = childkey.shape
    if len(key_shape) == 1:
        key_text = np.concatenate((childkey, ct_numbers[0:(len(ct_numbers) - key_shape[0])]))
    else:
        len_ct_numbers = len(ct_numbers)
        ct = np.zeros((29, len_ct_numbers), dtype=int)
        ct[:] = ct_numbers
        key_text = np.concatenate((childkey, ct[:, 0:(len(ct_numbers) - key_shape[1])]), axis=1)

    return np.subtract(ct_numbers, key_text) % 29


def calculate_fitness(childkey, ct_numbers, probabilities, algorithm, current_interrupter, reversed_text):
    mt = None
    if algorithm == 0:
        mt = decryption_vigenere(childkey, ct_numbers, current_interrupter)
    if algorithm == 1:
        mt = decryption_autokey(childkey, ct_numbers, current_interrupter)
    if algorithm == 2:
        mt = decryption_autokey_ciphertext(childkey, ct_numbers)
    if mt is None:
        raise AssertionError()

    if reversed_text:
        mt = mt[::-1]

    x = 29
    key_shape = childkey.shape
    if len(key_shape) == 1:
        x = 1
    score = np.zeros(x)
    for k in range(x):
        if x == 1:
            mt_slice = np.squeeze(mt)
        else:
            mt_slice = mt[k]
        indices = np.array([mt_slice[0:len(mt_slice) - 3] * 24389, mt_slice[1:len(mt_slice) - 2] * 841, mt_slice[2:len(mt_slice) - 1] * 29,
                            mt_slice[3:len(mt_slice)]])
        score[k] = np.sum(probabilities[np.sum(indices, axis=0)])
    return score


def translate_to_english(parent_key, reverse_gematria):
    dic = ["F", "U", "TH", "O", "R", "C", "G", "W", "H", "N", "I", "J", "EO", "P", "X", "S", "T", "B", "E", "M", "L",
           "ING", "OE", "D", "A", "AE", "Y", "IA", "EA"]
    if reverse_gematria:
        dic.reverse()
    translation = ""
    for index in np.nditer(parent_key):
        translation += dic[index]
    return translation


def translate_best_text(algorithm, best_key_ever, ct_numbers, current_interrupter, reverse_gematria):
    if algorithm == 0:
        return translate_to_english(decryption_vigenere(best_key_ever, ct_numbers, current_interrupter), reverse_gematria)
    if algorithm == 1:
        return translate_to_english(decryption_autokey(best_key_ever, ct_numbers, current_interrupter), reverse_gematria)
    if algorithm == 2:
        return translate_to_english(decryption_autokey_ciphertext(best_key_ever, ct_numbers), reverse_gematria)
    else:
        print('Invlaid algorithm ID')
