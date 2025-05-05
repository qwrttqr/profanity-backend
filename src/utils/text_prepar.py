from utils import deobfuscation_table_load, n_grams_load
from pymorphy2 import MorphAnalyzer
from nltk.stem.snowball import SnowballStemmer
import re


class TextPreparation:
    """
    Prepares text for the processing with the next pipeline step.
    Methods:
        prepare_text(text: str, word_basing_method: str = 'lemmatization', deobfuscation: bool = True): prepares text step by step.
        1 step - raw preparing(splitting by spaces, delete urls, replace ['Ё', 'ё'] -> e based on regexp, remove punctuation and special characters, lowecase all).
        2 step - deobfuscation(make replacements based on deobfuscation table(e.g. М4рия -> мария)). The replacements takes place in 2 stages:
            2.1 - non_single_char replacements. 
            2.2 - Unambiguous replacement(list lenght in table equal 1).
            2.3 - Ambiguous replacement(list lenght in table > 1).
            *stage 2.3 is happening with trying to find best n-gramm(n-grams with max lenght or n-grams with min occurence in case with equal n-gram lenght).
        3 step - schrinking(delete all 3 and more letters in row).
        4 step - clearing(delete all non-letters symbols left after deobfuscation).
        5 step - base word forming(based on word basing method).
    Warning - 3rd step is implemented by pymorphy2 and nltk snowballStemmer, see the docs for details 
        Returns list of deobfuscated lemmatized strings.
    """
    def __init__(self):
        self.__deobfuscation_table = deobfuscation_table_load()
        self.__morpher = MorphAnalyzer()
        self.__stemmer = SnowballStemmer(language = 'russian')
        self.__n_grams = n_grams_load()

    def prepare_text(self, text: str, word_basing_method: str = 'lemmatization', deobfuscation: bool = True, basing: bool = True):
        """
        Prepares the text.

        Parameters:
            text(str): text should be prepared.
            word_basing_method(str): which method of word basing we should use. Should be "lemmatization" or "stemming".
            deobfuscation(bool): optional, whether we do deobfuscation. True by default.
            basing(bool): optional, controls whether or not we doing lemmatization + stemming after text clearing. True by default.
        Returns:
            list: Words after prepating
        """
        pre_cleared = self.__raw_preparing(text)
        prepared_words = []
        for item in pre_cleared:
            deobfuscated = item

            if deobfuscation: # If we do deobfuscation
                deobfuscated = self.__deobfuscate(item)

            schrinked = self.__delete_long_vowels(deobfuscated)
            cleared = self.__get_letters_only(schrinked)
            cleared = list(cleared)

            while '' in cleared:
                cleared.remove('')

            cleared = ''.join(cleared)

            if basing:
                done_word = self.__get_base_form(''.join(cleared), word_basing_method)
            else:
                done_word = ''.join(cleared)

            prepared_words.append(done_word)

        return prepared_words


    def __deobfuscate(self, word: str):
        """
        Deobfuscates given word.

        Returns:
            str: Deobfuscated word.
        """

        deobfuscation_table_single = self.__deobfuscation_table['single_char_seq']
        deobfuscation_table_non_single = self.__deobfuscation_table['non_single_char_seq']

        for key,value in deobfuscation_table_non_single.items():
            word = word.replace(key, value[0])

        word = list(word)

        for i in range(len(word)):
            if word[i] in deobfuscation_table_single.keys():
                if len(deobfuscation_table_single[word[i]]) == 1:

                    value = deobfuscation_table_single[word[i]][0]
                    has_right_vowel = False
                    has_left_vowel  = False

                    if i > 0:
                        if word[i - 1].isalpha(): 
                            has_left_vowel = True

                    if i < len(word) - 1:
                        if word[i + 1].isalpha():
                            has_right_vowel = True

                    if has_left_vowel or has_right_vowel:
                        word[i] = word[i].replace(word[i], value)

        for i in range(len(word)):
            if word[i] in deobfuscation_table_single.keys():
                if len(deobfuscation_table_single[word[i]]) > 1:

                    values = deobfuscation_table_single[word[i]]
                    has_right_vowel = False
                    has_left_vowel  = False

                    if i > 0:
                        if word[i - 1].isalpha(): 
                            has_left_vowel = True

                    if i < len(word) - 1:
                        if word[i + 1].isalpha():
                            has_right_vowel = True

                    if has_left_vowel or has_right_vowel:
                        candidate = self.__find_best_n_gram(''.join(word), values, i)
                        if candidate:
                            word[i] = word[i].replace(word[i], candidate)

        word = ''.join(word)

        return word
    
    def __find_best_n_gram(self, word: str, candidates: list, position: int):
        """
        Finds best n_gram for given word.
        Best means longest. If 2 or more n_grams fit - n_gram with the lowest amount will be chosen.

        Returns:
            str/None: More suitable candidate or None if there is no possible candidate.
        """

        best_candidate = None
        best_lenght = 0
        min_count = float('inf')
        l = r = position

        for candidate in candidates:
            for l in range(position + 1):
                for r in range(position, len(word)):

                    n_gramma = word[l:position] + candidate + word[position + 1: r + 1]

                    if n_gramma in self.__n_grams.keys():
                        if len(n_gramma) > best_lenght:
                            best_candidate = candidate
                            best_lenght = len(n_gramma)

                        elif len(n_gramma) == best_lenght:
                            if self.__n_grams[n_gramma] < min_count:
                                min_count = self.__n_grams[n_gramma]
                                best_candidate = candidate

        return best_candidate

    def __raw_preparing(self, text: str):
        """
        Doing raw string preparing. Deletes url's, replaces letters, lowercasing all and splitting given text by spaces.
        Remove message footer(sign in yandex).
        Returns:
            list: Words array.
        """
        if ('--' in text):
            text = text[:text.rfind('--') + 1]
        text_string = text
        space_pattern = '\s+'
        line_break_pattern = '\n+'
        giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  

        # Regex to match most common email formats
        # See https://uibakery.io/regex-library/email-regex-python for detials (RFC 5322)
        email_regex = "(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21\\x23-\\x5b\\x5d-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\\x01-\\x08\\x0b\\x0c\\x0e-\\x1f\\x21-\\x5a\\x53-\\x7f]|\\\\[\\x01-\\x09\\x0b\\x0c\\x0e-\\x7f])+)\\])"

        # See https://uibakery.io/regex-library/email-regex-python
        parsed_text = re.sub(space_pattern, ' ', text_string)
        parsed_text = re.sub(line_break_pattern,' ', parsed_text)
        parsed_text = re.sub(giant_url_regex, '', parsed_text)
        parsed_text = re.sub(email_regex,'', parsed_text)
        parsed_text = parsed_text.replace('Ё', 'е')
        parsed_text = parsed_text.replace('ё', 'е')
        parsed_text = parsed_text.replace('_', ' ')
        parsed_text = parsed_text.replace('-', ' ')

        for item in ['/', '\\']:
            parsed_text = parsed_text.replace(item, '')

        parsed_text = parsed_text.lower()
        parsed_text = parsed_text.split()

        return parsed_text
    
    def __get_base_form(self, word: str, word_basing_method: str = "lemmatization"):
        """
        Returning base form for word with method based on word_basing_method param.

        Returns:
            str: Base form for word.
        """

        if word_basing_method == 'lemmatization':

            return self.__morpher.parse(word)[0].normal_form
        
        elif word_basing_method == 'stemming':
            word = self.__morpher.parse(word)[0].normal_form

            return self.__stemmer.stem(word)


    def __delete_long_vowels(self, text: str):
        """
        Handling 3 and more same letters in a row.

        Returns:
            str: Text with no more 3 same letter.
        """

        pattern = re.compile(r"(.)\1{2,}")
        new_text = pattern.sub(r"\1", text)

        return new_text

    def __get_letters_only(self, word: str):
        """
        Selects only letters from given word.

        Returns:
            str: Word only with letters.
        """
        return re.sub(r'[^а-яё]', '', word)