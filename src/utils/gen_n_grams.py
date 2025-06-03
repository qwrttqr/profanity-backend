import os
import json

n_gramms = {}


def generate_n_grams():
    if os.path.exists('./files'):
        try:
            with open('./files/words.txt', encoding='utf-8') as words:
                for word in words:
                    for l in range(len(word)-1):
                        for r in range(l + 1, len(word)):
                            n_gramm = word[l:r + 1]
                            if n_gramm not in n_gramms.keys():
                                n_gramms[n_gramm] = 1
                            else:
                                n_gramms[n_gramm] += 1
                with open('./files/n_grams.json', 'w+') as file:
                    json.dump(n_gramms, file)
                    print('Successful generation')
        except:
            print('Error during trying to touch n_grams file, check files folder')
            raise Exception
