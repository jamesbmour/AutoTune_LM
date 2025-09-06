import json
import re
from typing import List, Dict, Any

from qa_generation.ollama_client import OllamaClient
from qa_generation.prompt_templates import PromptTemplates

class QAGenerator:
    """
    Generates question-answer pairs from text chunks using an LLM.
    """

    def __init__(self, config: Dict[str, Any], prompts: PromptTemplates):
        """
        Initializes the QAGenerator.

        Args:
            config (Dict[str, Any]): The configuration dictionary.
            prompts (PromptTemplates): The prompt templates.
        """
        self.config = config
        self.prompts = prompts
        self.llm_client = OllamaClient()

    def generate_qa_pairs(self, chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Generates Q&A pairs for a list of text chunks.

        Args:
            chunks (List[Dict[str, str]]): A list of text chunks.

        Returns:
            List[Dict[str, str]]: A list of generated Q&A pairs.
        """
        qa_pairs = []
        for chunk in chunks:
            structured_chunk = self._structure_chunk(chunk)
            if structured_chunk:
                generated_pairs = self._generate_pairs_for_chunk(structured_chunk)
                if generated_pairs:
                    qa_pairs.extend(generated_pairs)
        return qa_pairs

    def _structure_chunk(self, chunk: Dict[str, str]) -> str | None:
        """
        Structures a text chunk using the LLM.

        Args:
            chunk (Dict[str, str]): The text chunk to structure.

        Returns:
            str: The structured text chunk, or None if an error occurred.
        """
        prompt = self.prompts.get_prompt('structuring_prompt', text_chunk=chunk['content'])
        response = self.llm_client.generate(model=self.config['llm_model'], prompt=prompt)
        if "error" in response:
            return None
        return response.get('response', '')

    def _generate_pairs_for_chunk(self, structured_chunk: str) -> List[Dict[str, str]] | None:
        """
        Generates Q&A pairs for a single structured chunk.

        Args:
            structured_chunk (str): The structured text chunk.

        Returns:
            List[Dict[str, str]]: A list of generated Q&A pairs, or None if an error occurred.
        """
        prompt = self.prompts.get_prompt('qa_generation_prompt', structured_chunk=structured_chunk)
        response = self.llm_client.generate(model=self.config['llm_model'], prompt=prompt)
        if "error" in response:
            return None
        response_text = response.get('response', '')
        
        # Use regex to find all JSON objects in the response
        json_objects = re.findall(r'\{.*?\}', response_text)
        
        pairs = []
        for obj in json_objects:
            try:
                pairs.append(json.loads(obj))
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON: {obj}")
        return pairs
