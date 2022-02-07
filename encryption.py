import numpy as np
import prime_array as pa


def encrypt_text(plaintext, algorithm, shift_id, reverse_text, reverse_gematria, interrupter, key, interrupter_array):
    if reverse_text:
        plaintext = plaintext[::-1]

    # What happens with interrupters at this point?
    if reverse_gematria:
        plaintext = atbash(plaintext)

    ct = plaintext.copy()
    ct = apply_shift(ct, interrupter, shift_id)

    if algorithm.lower() == 'vigenere':
        ct = vigenere_encryption(ct, interrupter, key, interrupter_array)
    elif algorithm.lower() == 'autokey':
        ct = autokey_encryption(ct, interrupter, key, interrupter_array)
    else:
        assert False
    return ct


def vigenere_encryption(ct, interrupter, key, interrupter_array=None):
    key_index = 0
    key_length = len(key)

    if interrupter_array is None:
        interrupter_array = np.asarray(ct == interrupter, dtype=bool)

    for index, element in enumerate(ct):
        if interrupter_array[index]:
            continue
        ct[index] = (ct[index] + key[key_index % key_length]) % 29
        key_index += 1
    return ct


def autokey_encryption(ct, interrupter, key, interrupter_array=None):
    plaintext = ct.copy()
    counter = 0
    index = 0

    if interrupter_array is None:
        interrupter_array = np.asarray(ct == interrupter, dtype=bool)

    while counter < len(key):
        if interrupter_array[index]:
            index += 1
            continue
        ct[index] = (plaintext[index] + key[counter]) % 29
        index += 1
        counter += 1

    position = 0
    for i in range(index, len(ct)):
        if interrupter_array[i]:
            continue
        ct[i] = (ct[i] + plaintext[position]) % 29
        position += 1

    return ct


def translate_to_english(input_text, reverse_gematria):
    dic = ["F", "U", "TH", "O", "R", "C", "G", "W", "H", "N", "I", "J", "EO", "P", "X", "S", "T", "B", "E", "M", "L",
           "ING", "OE", "D", "A", "AE", "Y", "IA", "EA"]
    if reverse_gematria:
        dic.reverse()
    translation = ""
    for index in np.nditer(input_text):
        translation += dic[index]
    return translation


def atbash(plaintext):
    return np.remainder(28 - plaintext, 29)


def apply_shift(ct_numbers, interrupter, shift_id):
    # Indices 1-2, 3-4, and 5-6 are swapped, so that decryption undoes the encryption operation.
    # So Index 1 in encryption is the inverse operation of Index 1 in decryption
    shift_index = 0
    prime = pa.numpy_prime_array()

    if shift_id == 1:
        for index, element in enumerate(ct_numbers):
            if element == interrupter:
                continue
            ct_numbers[index] -= prime[shift_index] - 1
            shift_index += 1
    elif shift_id == 2:
        for index, element in enumerate(ct_numbers):
            if element == interrupter:
                continue
            ct_numbers[index] += prime[shift_index] - 1
            shift_index += 1
    elif shift_id == 3:
        for index, element in enumerate(ct_numbers):
            if element == interrupter:
                continue
            ct_numbers[index] -= prime[shift_index]
            shift_index += 1
    elif shift_id == 4:
        for index, element in enumerate(ct_numbers):
            if element == interrupter:
                continue
            ct_numbers[index] += prime[shift_index]
            shift_index += 1
    elif shift_id == 5:
        for index, element in enumerate(ct_numbers):
            if element == interrupter:
                continue
            ct_numbers[index] -= shift_index
            shift_index += 1
    elif shift_id == 6:
        for index, element in enumerate(ct_numbers):
            if element == interrupter:
                continue
            ct_numbers[index] += shift_index
            shift_index += 1
    return np.remainder(ct_numbers, 29)


def convert_text_to_index(input_text):
    latin_fragments = ['F', 'U', 'TH', 'O', 'R', 'C', 'G', 'W', 'H', 'N', 'I', 'J', 'EO', 'P', 'X', 'S', 'T', 'B', 'E', 'M', 'L', 'ING',
                       'OE',
                       'D', 'A', 'AE', 'Y', 'IA', 'EA']

    e2p = {}
    for i in range(0, 29):
        e2p[latin_fragments[i]] = i
    e2p["IO"] = e2p["IA"]
    e2p["K"] = e2p["C"]
    e2p["NG"] = e2p["ING"]
    e2p["Z"] = e2p["S"]
    e2p["Q"] = e2p["C"]
    e2p["V"] = e2p["U"]

    input_text = input_text.upper().replace("QU", "KW")
    input_text = input_text.replace("Q", "K")

    text_as_index = []
    skip = 0
    for index, value in enumerate(input_text):
        if skip:
            skip -= 1
            continue
        if input_text[index:index + 3] in e2p:
            text_as_index.append(e2p[input_text[index:index + 3]])
            skip = 2
            continue
        elif input_text[index:index + 2] in e2p:
            text_as_index.append(e2p[input_text[index:index + 2]])
            skip = 1
        else:
            text_as_index.append(e2p[input_text[index]])
    return np.asarray(text_as_index)
