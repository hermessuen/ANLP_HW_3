import numpy as np
import os
import pandas as pd
import argparse

_Delimiter = {0: '?', 1: '!'}
p = 0.1

def read_file(fn):
    with open(fn) as file:
        lines = [line.rstrip() for line in file]
    return lines

def write_file(fn, data):
    with open(fn, 'w') as file:
        for d in data:
            file.writelines(d + '\n')


def inject_question_mark(data):
    ret = []
    for item in data:
        sentence, label, c = item.split("\t")
        last = sentence[-1]
        swap = np.random.binomial(1, p)
        if swap:
            rand_d = _Delimiter[np.random.binomial(1, 0.5)]
            sent = sentence if last.isalnum() else sentence[:-1]
            print(f"Swapping: [[{sentence}]] with =={sent + rand_d}==") 
            sentence = sent + rand_d
            
        ret += ["\t".join([sentence, label, c])]
            
    return ret


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input-path", type=str, help="data file path", default="./data/original/train.tsv")
    parser.add_argument("--save-path", type=str, help="save data file path", default="./data/original/train_adv.tsv")

    args = parser.parse_args()
    
    data_fn = args.input_path
    data = read_file(data_fn)
    ret = inject_question_mark(data)
    save_fn = args.save_path
    write_file(save_fn, ret)    