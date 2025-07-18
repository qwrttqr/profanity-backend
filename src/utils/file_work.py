import os
import json
from typing import Any, Callable
from pathlib import Path


class FileManager:
    FileManagerInstance = None

    def __new__(cls, *args, **kwargs):
        if cls.FileManagerInstance is None:
            cls.FileManagerInstance = super().__new__(cls)
        return cls.FileManagerInstance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.__CONFIG_PATH = self.__get_project_root() / 'config.json'
            self.__check_models_folder_exist()
            self.__check_n_grams_file()
            self.__check_config_file()
            self.config_file = self.load_file((self.__get_project_root() / 'config.json'), json.load, 'r', False)
            self.n_grams_file = self.load_file(self.__get_project_root() / 'files' / 'n_grams.json', json.load,
                                               'r', False)
            self.deobfuscation_table_file = self.load_file(self.__get_project_root() / 'files' /
                                                           'deobfuscation_table.json', json.load, 'r', False)

    def __check_n_grams_file(self):
        if not (os.path.exists(self.__get_project_root() / 'files' / 'n_grams.json')):
            print('n-grams file not found')
            self.__generate_n_grams()

    def __check_models_folder_exist(self):
        if not (os.path.exists(self.__get_project_root() / 'models' / 'profanity_model')):
            print('Profanity model path is not exist')
            self.__create_profanity_dir()
            print('Profanity path created')
        if not (os.path.exists(self.__get_project_root() / 'models' / 'semantic_model')):
            print('Semantic model path is not exist')
            self.__create_semantic_dir()
            print('Semantic path created')

    def __check_config_file(self):
        if not (os.path.exists(self.__get_project_root() / 'config.json')):
            print('Config file is not exists, creating')

    @staticmethod
    def __get_project_root() -> Path:
        """
        Returns the absolute path to the project root directory

        Returns:
            Path - project root path
        """

        return Path(__file__).parent.parent.parent

    def __generate_n_grams(self):
        self.n_gramms = {}
        if os.path.exists('./files'):
            try:
                with open('./files/words.txt', encoding='utf-8') as words:
                    for word in words:
                        for l in range(len(word) - 1):
                            for r in range(l + 1, len(word)):
                                n_gramm = word[l:r + 1]
                                if n_gramm not in self.n_gramms.keys():
                                    self.n_gramms[n_gramm] = 1
                                else:
                                    self.n_gramms[n_gramm] += 1
                    with open('./files/n_grams.json', 'w+') as file:
                        json.dump(self.n_gramms, file)
            except:
                print('Error during trying to touch n_grams file, check files folder')
                raise Exception

    def __create_profanity_dir(self):
        os.makedirs(self.__get_project_root() / 'models' / 'profanity_model')

    def __create_semantic_dir(self):
        os.makedirs(self.__get_project_root() / 'models' / 'semantic_model')

    def __create_config_file(self):
        with open(self.__get_project_root() / 'config.json', 'w+') as fl:
            startup_conf = {
                "profanity_model": {
                    "ver": None
                },
                "semantic_model": {
                    "ver": None
                }
            }
            json.dump(startup_conf, fl, ensure_ascii=False, indent=2)

    def load_file(self,
        file_path: Path,
        loader: Callable,
        file_type: str = 'file',
        binary: bool = False
    ):
        """
        Generic file loader with error handling.

        Args:
            file_path: path to the file
            loader: function to load the file (joblib.load/json.load)
            file_type: description of file type for error messages
            binary: whether to open in binary mode

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

        except FileNotFoundError as e:
            print(f'File not found {str(e)}')
        except json.JSONDecodeError:
            raise ValueError(f'Invalid JSON format in {file_type} file')
        except Exception as e:
            raise RuntimeError(f'Error loading {file_type}: {str(e)}')

    def save_file(self,
        file_path: Path,
        save_item: Any,
        save_func: Callable[[Any, Path], None],
        **kwargs
    ) -> None:
        """
        Args:
            file_path: Path - path to file
            save_item: Any - item to be saved
            save_func: Callable - function like `joblib.dump` or `model.save_pretrained`

        """
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if save_func.__name__ == "save_pretrained":  # Hugging Face case
                save_func(file_path, **kwargs)
            elif save_func.__name__ == "joblib.dump":
                save_func(save_item, file_path)
            elif save_func.__name__ == "json.dump":
                save_func(save_item, **kwargs)
        except Exception as e:
            print(f"Error saving to {file_path}: {str(e)}")
            raise

    def get_semantic_ver(self) -> int:
        try:
            return self.config_file['semantic_model']['ver']
        except Exception as e:
            print(f'Error during configuration semantic loading: {str(e)}')
            raise Exception('Error during configuration semantic loading')

    def get_profanity_ver(self) -> int:
        try:
            return self.config_file['profanity_model']['ver']
        except Exception as e:
            print(f'Error during configuration semantic loading: {str(e)}')
            raise Exception('Error during configuration semantic loading')

    def get_profanity_model_path(self, ver) -> Path:
        """
        Returns path by giver profanity_model ver
        Args:
            ver: int - profanity model ver

        Returns:
            Path - path to profanity model by given ver
        """
        return self.__get_project_root() / 'models' / 'profanity_model' / f'ver{ver}'

    def get_semantic_model_path(self, ver) -> Path:
        """
        Returns path by giver profanity_model ver
        Args:
            ver: int - profanity model ver

        Returns:
            Path - path to profanity model by given ver
        """
        return self.__get_project_root() / 'models' / 'semantic_model' / f'ver{ver}'

    def get_deobfuscation_table(self) -> dict[str, Any]:

        return self.deobfuscation_table_file

    def get_n_grams_file(self) -> dict[str, str]:

        return self.n_grams_file

    def save_profanity_info(self, ver, model_data):
        """
        Saves profanity model info into config and folder with version.
        Args:
            ver: int - version of model
            model_data: data about model(learning date, ver, etc.)

        Raises:
            Exception('Error during configuration profanity saving {e}')

        """
        try:
            with open(self.__CONFIG_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)

            data["profanity_model"]["ver"] = ver

            with open(self.__CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            with open(self.get_profanity_model_path(ver) / 'model_info.json', 'w+', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f'Error during configuration profanity saving: {str(e)}')
            raise Exception(f'Error during configuration profanity saving: {str(e)}')

    def save_semantic_info(self, ver, model_data):
        try:
            with open(self.__CONFIG_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)

            data["semantic_model"]["ver"] = ver

            with open(self.__CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            with open(self.get_semantic_model_path(ver) / 'model_info.json', 'w+', encoding='utf-8') as f:
                json.dump(model_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f'Error during configuration semantic loading: {str(e)}')
            raise Exception('Error during configuration semantic saving')
