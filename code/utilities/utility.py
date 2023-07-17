import numpy as np
import pickle
import random
import os


def ones_to_str(ones: set, n: int)->str:
    """
    Function to transform the set of positions of '1's into the corresponding binary string.

    Arguments:
        - ones: set, the set of positions
        - n: int, the length of the resulting binary string

    Returns:
        The binary string of length n where the '1's are at the positions collected in ones
    """
    binary = ["0" for i in range(n)]
    for i in ones:
        binary[i] = "1"
    return "".join(binary)


def str_to_ones(string: str)->set:
    """
    Function to transform a binary string into a set collecting the positions of the '1's.

    Arguments:
        - string: str, the binary string

    Returns:
        The set collecting the positions of the '1's in string
    """
    ones = [i for i in range(len(string)) if string[i] == "1"]
    return set(ones)


def generate_functions(n: int, number: int):
    if number <= 0:
        raise ValueError("number must be > 0")
    functions = []
    while len(functions) < number:
        f = ""
        for i in range(2**n):
            f += str(random.randint(0, 1))
        if f not in functions:
            functions.append(f)
    return functions


def get_functions(n: int, number: int):
    """
    Function to save a set of target concepts in a file, if this file exists, retrieve the functions.

    Arguments:
        - n: int, the dimension of the input space
        - number: int (default=0), the number of functions in the set. If 0, then it is the whole class of concepts

    Returns:
        - A list containing the functions
    """
    if number > 2**(2**n) or number <= 0:
        raise ValueError(f"number should be 0 < number <= {2**(2**n)}")

    directory = f"{os.getcwd()}/functions"

    if not os.path.exists(directory):
        os.makedirs(directory)

    file = f"{directory}/functions_{n}_{number}.txt"

    if os.path.exists(file):
        with open(file, "rb") as f:
            functions = pickle.load(f)
    else:
        functions = generate_functions(n, number)

        with open(file, "wb") as f:
            pickle.dump(functions, f)

    return functions


def get_permutation(n: int):
    p = [[] for _ in range(n+1)]
    permut = []

    for i in range(2**n):
        b = format(i, f"0{n}b")
        ones = str_to_ones(b)
        p[len(ones)].append(int(b,2))

    for g in p:
        permut += g

    return permut


def collect(data_directory, metric):
    collection = {}
    files = []

    for f in os.listdir(data_directory):
        if metric in f:
            files.append(f)
    
    with open(f"{data_directory}/{files[0]}", "rb") as f:
        data = pickle.load(f)
    for k in data:
        collection[k] = [data[k]]

    for i in range(1,len(files)):
        with open(f"{data_directory}/{files[i]}", "rb") as f:
            data = pickle.load(f)
        for k in data:
            collection[k].append(data[k])
    
    return collection



