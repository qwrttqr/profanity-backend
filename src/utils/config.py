import json
from src.utils.load import files, config_path



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

    except Exception as e:
        print(f'Error during configuration semantic loading: {str(e)}')
        raise Exception('Error during configuration semantic saving')
