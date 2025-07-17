import os
import json
from pathlib import Path
from .gen_n_grams import generate_n_grams


def get_project_root() -> Path:
    """
    Returns the absolute path to the project root directory
    """

    return Path(__file__).parent.parent.parent


def load_file(
    file_path: Path,
    loader: callable,
    file_type: str = 'file',
    binary: bool = False
):
    """
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
    """

    try:
        mode = 'rb' if binary else 'r'
        encoding = None if binary else 'utf-8'

        with open(file_path, mode, encoding=encoding) as f:

            return loader(f)

    except FileNotFoundError:
        if file_path == n_grams_path:
            print('n-gramm file not found, generating...')
            generate_n_grams()
        elif file_path == profanity_path:
            print('profanity models file not found')
            check_models_folder_exist()
    except json.JSONDecodeError:
        raise ValueError(f'Invalid JSON format in {file_type} file')
    except Exception as e:
        raise RuntimeError(f'Error loading {file_type}: {str(e)}')


config_path = get_project_root() / 'config.json'
n_grams_path = get_project_root() / 'files' / 'n_grams.json'
deobfuscation_table_path = get_project_root() / 'files' / \
                           'deobfuscation_table.json'
profanity_path = get_project_root() / 'models' / 'profanity_model'
semantic_directory_path = (get_project_root() / 'models' / 'semantic_model')

files = {
    'n-grams': None,
    'deobfuscation_table': None,
    'config': None
}

files['n-grams'] = load_file(n_grams_path, json.load, 'r', False)
files['deobfuscation_table'] = load_file(
    deobfuscation_table_path,
    json.load, 'r', False)
files['config'] = load_file(config_path, json.load, 'r', False)

profanity_model_info_path = get_project_root() / 'models' / 'profanity_model' / f"ver{files['config']['profanity_model']['ver']}" / 'model_info.json'
semantic_model_info_path = get_project_root() / 'models' / 'semantic_model' / f"ver{files['config']['semantic_model']['ver']}" / 'model_info.json'
files['profanity_model_info'] = load_file(profanity_model_info_path, json.load, 'r', False)
files['semantic_model_info'] = load_file(semantic_model_info_path, json.load, 'r', False)


def create_profanity_dir():
    os.makedirs(get_project_root() / 'models' / 'profanity_model')


def create_semantic_dir():
    os.makedirs(get_project_root() / 'models' / 'semantic_model')


def check_models_folder_exist():
    if not (os.path.exists(get_project_root() / 'models' / 'profanity_model')):
        print('Profanity model path is not exist')
        create_profanity_dir()
        print('Profanity path created')
    if not (os.path.exists(get_project_root() / 'models' / 'semantic_model')):
        print('Semantic model path is not exist')
        create_semantic_dir()
        print('Semantic path created')


def get_profanity_ver():
    try:

        return files['config']['profanity_model']['ver']
    except Exception as e:
        print(f'Error during configuration profanity loading: {str(e)}')
        raise Exception('Error during configuration profanity loading')


def get_semantic_ver():
    try:
        return files['config']['semantic_model']['ver']
    except Exception as e:
        print(f'Error during configuration semantic loading: {str(e)}')
        raise Exception('Error during configuration semantic loading')


def save_profanity_info(ver, model_data, model_dir):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data["profanity_model"]["ver"] = ver

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        with open(model_dir, 'w+', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

        files['profanity_model_info'] = load_file(profanity_model_info_path, json.load, 'r', False)

    except Exception as e:
        print(f'Error during configuration profanity saving: {str(e)}')
        raise Exception('Error during configuration profanity saving')


def save_semantic_info(ver, model_data, model_dir):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        data["semantic_model"]["ver"] = ver

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        with open(model_dir, 'w+', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)

        files['semantic_model_info'] = load_file(semantic_model_info_path, json.load, 'r', False)

    except Exception as e:
        print(f'Error during configuration semantic loading: {str(e)}')
        raise Exception('Error during configuration semantic saving')
