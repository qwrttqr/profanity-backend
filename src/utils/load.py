import joblib
import json
from .gen_n_grams import generate_n_grams
from pathlib import Path


def get_project_root() -> Path:
    '''Returns the absolute path to the project root directory'''

    return Path(__file__).parent.parent.parent


def load():
    '''
    Loads the vectorizer model from models/ directory.
    Loads the ML model from models / directory.
    Loads n-grams data from files/ directory.
    Loads deobfuscation table from files / directory.
    '''

    n_grams_path = get_project_root() / 'files' / 'n_grams.json'
    vectorizer_path = get_project_root() / 'models' / 'vectorizer.joblib'
    model_path = get_project_root() / 'models' / 'model.joblib'
    deobfuscation_table_path = get_project_root() / 'files' / \
        'deobfuscation_table.json'

    files = {
        'vectorizer_model': None,
        'ML_model': None,
        'n-grams': None,
        'deobfuscation_table': None
    }

    try:
        with open(vectorizer_path, 'rb') as f:  # Note 'rb' mode for binary files
            files['vectorizer_model'] = joblib.load(f)

    except FileNotFoundError:
        raise FileNotFoundError(
            f'Vectorizer file not found at: {vectorizer_path}')

    except Exception as e:
        raise RuntimeError(f'Error loading vectorizer: {str(e)}')


    try:
        with open(model_path, 'rb') as f:  # Note 'rb' mode for binary files
            files['ML_model'] = joblib.load(f)

    except FileNotFoundError:
        raise FileNotFoundError(f'Model file not found at: {model_path}')

    except Exception as e:
        raise RuntimeError(f'Error loading model: {str(e)}')


    try:
        with open(n_grams_path, 'r', encoding='utf-8') as f:
            files['n-grams'] = json.load(f)

    except FileNotFoundError:

        print('n-grams file not found. Started generation....')
        generate_n_grams()

    except json.JSONDecodeError:
        raise ValueError('Invalid JSON format in n_grams file')

    except Exception as e:
        raise RuntimeError(f'Error loading n_grams: {str(e)}')
    

    try:
        with open(deobfuscation_table_path, 'r', encoding='utf-8') as f:
            files['deobfuscation_table'] = json.load(f)

    except FileNotFoundError:
        raise FileNotFoundError(
            f'deobfuscation_table file not found at: {deobfuscation_table_path}')

    except json.JSONDecodeError:
        raise ValueError('Invalid JSON format in deobfuscation_table file')

    except Exception as e:
        raise RuntimeError(f'Error loading deobfuscation_table: {str(e)}')

    return files


files = load()