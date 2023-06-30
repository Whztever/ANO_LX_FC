import configparser
import os
from typing import Any, Optional

import numpy as np


class ConfigManager:
    def __init__(
        self,
        file: str = "config.ini",
        default_setting: dict = {},
        section: str = "SETTINGS",
    ):
        """
        file: config file path
        default_setting: default settings dictionary, set if the option is not set yet
        section: section name
        """
        path = os.path.dirname(__file__)
        file_path = os.path.join(path, file)
        self._config_file = file_path
        self._config = configparser.ConfigParser()
        if os.path.exists(file_path):
            self._config.read(file_path)
        else:
            open(file_path, "w").close()
        self._section_name = section.upper()
        self._init_file(default_setting.copy())

    def set(self, option: str, value: Any):
        """
        Set the value of an option
        """
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, str):
            value = repr(value)
        self._config.set(self._section_name, option, value)
        with open(self._config_file, "w") as f:
            self._config.write(f)

    def remove(self, option: str):
        """
        Remove an option
        """
        self._config.remove_option(self._section_name, option)
        with open(self._config_file, "w") as f:
            self._config.write(f)

    def _init_file(self, default_setting: dict):
        if self._section_name not in self._config.sections():
            self._config.add_section(self._section_name)

        for key, value in default_setting.items():
            if not self._config.has_option(self._section_name, key):
                self.set(key, value)

    def set_from_dict(self, setting_dict: dict):
        """
        Set values by a given dictionary
        """
        for key, value in setting_dict.items():
            self.set(key, value)

    def dict(self) -> dict[str, str]:
        """
        Return a dictionary of all options
        """
        items = self._config.items(self._section_name)
        return {i[0]: i[1] for i in items}

    def clear_all(self):
        """
        Remove all options
        """
        for key in self.dict().keys():
            self._config.remove_option(self._section_name, key)
        with open(self._config_file, "w") as f:
            self._config.write(f)

    def get(self, option: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get the value of an option, return as string
        """
        if self._config.has_option(self._section_name, option):
            return self._config.get(self._section_name, option)
        else:
            if default is not None:
                self.set(option, default)
            return default

    def get_bool(self, option: str, default: Optional[bool] = None) -> Optional[bool]:
        """
        Get the value of an option, return as boolean
        """
        if self._config.has_option(self._section_name, option):
            return self._config.getboolean(self._section_name, option)
        else:
            if default is not None:
                self.set(option, default)
            return default

    def get_int(self, option: str, default: Optional[int] = None) -> Optional[int]:
        """
        Get the value of an option, return as integer
        """
        if self._config.has_option(self._section_name, option):
            return self._config.getint(self._section_name, option)
        else:
            if default is not None:
                self.set(option, default)
            return default

    def get_float(self, option: str, default: Optional[float] = None) -> Optional[float]:
        """
        Get the value of an option, return as float
        """
        if self._config.has_option(self._section_name, option):
            return self._config.getfloat(self._section_name, option)
        else:
            if default is not None:
                self.set(option, default)
            return default

    def get_eval(self, option: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Get the value of an option, return as python object
        """
        if self._config.has_option(self._section_name, option):
            return eval(self._config.get(self._section_name, option))
        else:
            if default is not None:
                self.set(option, default)
            return default

    def get_array(
        self, option: str, dtype: Optional[str] = None, default: Optional[Any] = None
    ) -> Optional[np.ndarray]:
        """
        Get the value of an option, return as numpy array
        dtype: str, optional, numpy format, default None
        """
        if self._config.has_option(self._section_name, option):
            string = self._config.get(self._section_name, option).strip()
            # string = re.sub(r"\[\s+", r"[", string)
            # string = re.sub(r"\s(\d)", r",\1", string)
            # string = string.replace("\n", ",")
            if dtype == None:
                return np.array(eval(string))
            else:
                return np.array(eval(string), dtype)
        else:
            if default is not None:
                self.set(option, default)
            return default
