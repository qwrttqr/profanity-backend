import joblib
import json
from .gen_n_grams import generate_n_grams
from pathlib import Path


def get_project_root() -> Path:
    '''Returns the absolute path to the project root directory'''

    return Path(__file__).parent.parent.parent


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


def load_file(
    file_path: Path,
    loader: callable,
    file_type: str = 'file',
    binary: bool = False
):
    '''
    Generic file loader with error handling.

    Args:
        file_path: Path to the file
        loader: Function to load the file (joblib.load/json.load)
        file_type: Description of file type for error messages
        binary: Whether to open in binary mode

    Returns:
        Loaded file content

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: For JSON decode errors
        RuntimeError: For other loading errors
    '''

    try:
        mode = 'rb' if binary else 'r'
        encoding = None if binary else 'utf-8'

        with open(file_path, mode, encoding=encoding) as f:
            return loader(f)

    except FileNotFoundError:
        raise FileNotFoundError(f'{file_type} file not found at: {file_path}')
    except json.JSONDecodeError:
        raise ValueError(f'Invalid JSON format in {file_type} file')
    except Exception as e:
        raise RuntimeError(f'Error loading {file_type}: {str(e)}')


files['vectorizer_model'] = load_file(vectorizer_path, joblib.load, 'rb', True)
files['ML_model'] = load_file(model_path, joblib.load, 'rb', True)
files['n-grams'] = load_file(n_grams_path, json.load, 'r', False)
files['deobfuscation_table'] = load_file(
    deobfuscation_table_path, json.load, 'r', False)
