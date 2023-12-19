from multiprocessing import Pool
import os
import sys
import random
from papote.bpe import BPE, chunks, clean_private_unicode


def count(args):
    filenames, bpe = args
    total = 0
    for i, fn in enumerate(filenames):
        with open(fn, errors='ignore') as f:
            text = clean_private_unicode(f.read())
            total += len(bpe.encode_text(text))
    return total


if __name__ == '__main__':
    directory = sys.argv[1]
    num_threads = 16
    files = [
        os.path.join(root, file) for root, dirs, files in os.walk(directory)
        for file in files
    ]
    random.shuffle(files)
    bpe = BPE.load(sys.argv[2])
    if num_threads == 1:
        count((files, bpe))
    else:
        with Pool(num_threads) as pool:
            total = 0
            for i, subtotal in enumerate(
                    pool.imap_unordered(
                        count,
                        ((chunk, bpe)
                         for chunk in chunks([file
                                              for file in files], num_threads)
                         ))):
                total += subtotal
            print(i, round(total / 1e6, 2), 'M')
        print('chinchilla optimal model size:', round(total / 20e6, 2), 'M')
