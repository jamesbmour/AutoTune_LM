import ollama
import os
from typing import Dict, Any

class OllamaClient:
    """
    A wrapper for the Ollama API client.
    """

    def __init__(self, host: str = None):
        """
        Initializes the Ollama client.

        Args:
            host (str, optional): The Ollama host. Defaults to None, which uses the OLLAMA_HOST environment variable.
        """
        self.client = ollama.Client(host=host or os.getenv("OLLAMA_HOST"))

    def has_model(self, model_name: str) -> bool:
        """
        Checks if a model is available in the Ollama instance.

        Args:
            model_name (str): The name of the model.

        Returns:
            bool: True if the model is available, False otherwise.
        """
        try:
            self.client.show(model_name)
            return True
        except Exception as e:
            return False

    def generate(self, model: str, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Generates a response from the Ollama model.

        Args:
            model (str): The name of the model to use.
            prompt (str): The prompt to send to the model.
            **kwargs: Additional arguments to pass to the ollama client.

        Returns:
            Dict[str, Any]: The response from the model.
        """
        try:
            response = self.client.generate(model=model, prompt=prompt, **kwargs)
            return response
        except Exception as e:
            # Add logging here
            print(f"Error generating response from Ollama: {e}")
            return {"error": str(e)}

if __name__ == '__main__':
    # Example usage
    client = OllamaClient()
    model = "llama3.2"  # Make sure this model is pulled in Ollama
    prompt = "Why is the sky blue?"
    response = client.generate(model=model, prompt=prompt)
    if "error" not in response:
        print(response.get("response"))

