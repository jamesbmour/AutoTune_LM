import yaml
from typing import Dict, Any

class PromptTemplates:
    """
    Loads and manages prompt templates from a YAML file.
    """

    def __init__(self, prompts_file: str):
        """
        Initializes the PromptTemplates by loading from a YAML file.

        Args:
            prompts_file (str): The path to the prompts YAML file.
        """
        self.prompts = self._load_prompts(prompts_file)

    def _load_prompts(self, prompts_file: str) -> Dict[str, Any]:
        """
        Loads prompts from a YAML file.

        Args:
            prompts_file (str): The path to the prompts YAML file.

        Returns:
            Dict[str, Any]: The loaded prompts.
        """
        with open(prompts_file, 'r') as f:
            return yaml.safe_load(f)

    def get_prompt(self, name: str, **kwargs: Any) -> str:
        """
        Gets a formatted prompt by name.

        Args:
            name (str): The name of the prompt template.
            **kwargs: The variables to format the prompt with.

        Returns:
            str: The formatted prompt.
        """
        template = self.prompts.get(name)
        if not template:
            raise ValueError(f"Prompt template '{name}' not found.")
        return template.format(**kwargs)

if __name__ == '__main__':
    # Example usage
    prompts = PromptTemplates(prompts_file='../../configs/prompts.yaml')
    structure_prompt = prompts.get_prompt('structure_prompt', text_chunk='This is a test chunk.')
    print(structure_prompt)
    qa_prompt = prompts.get_prompt('qa_generation_prompt', structured_chunk='This is a structured chunk.')
    print(qa_prompt)
