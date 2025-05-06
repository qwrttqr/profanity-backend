import joblib
import json
from .gen_n_grams import generate_n_grams
from pathlib import Path


def get_project_root():
    """Returns the absolute path to the project root directory"""

    return Path(__file__).parent.parent.parent

def vectorizer_load():
    """Loads the vectorizer model from models/ directory"""
    vectorizer_path = get_project_root() / 'models' / 'vectorizer.joblib'
    try:
        with open(vectorizer_path, 'rb') as f:  # Note 'rb' mode for binary files
            
            return joblib.load(f)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Vectorizer file not found at: {vectorizer_path}")
    
    except Exception as e:
        raise RuntimeError(f"Error loading vectorizer: {str(e)}")

def model_load():
    """Loads the ML model from models/ directory"""
    model_path = get_project_root() / 'models' / 'model.joblib'
    try:
        with open(model_path, 'rb') as f:  # Note 'rb' mode for binary files

            return joblib.load(f)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

def n_grams_load():
    """Loads n-grams data from files/ directory"""
    n_grams_path = get_project_root() / 'files' / 'n_grams.json'
    try:
        with open(n_grams_path, 'r', encoding='utf-8') as f:

            return json.load(f)
        
    except FileNotFoundError:

        print('n-grams file not found. Started generation....')
        generate_n_grams()
    
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in n_grams file")
    
    except Exception as e:
        raise RuntimeError(f"Error loading n_grams: {str(e)}")
    
def deobfuscation_table_load():
    """Loads deobfuscation table from files / directory"""
    n_grams_path = get_project_root() / 'files' / 'deobfuscation_table.json'
    try:
        with open(n_grams_path, 'r', encoding='utf-8') as f:

            return json.load(f)
        
    except FileNotFoundError:
        raise FileNotFoundError(f"deobfuscation_table file not found at: {n_grams_path}")
    
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in deobfuscation_table file")
    
    except Exception as e:
        raise RuntimeError(f"Error loading deobfuscation_table: {str(e)}")