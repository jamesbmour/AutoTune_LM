import yaml
from typing import Dict, Any

class ConfigLoader:
    """
    Loads configuration from a YAML file.
    """

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Loads a YAML configuration file.

        Args:
            config_path (str): The path to the configuration file.

        Returns:
            Dict[str, Any]: The loaded configuration.
        """
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
