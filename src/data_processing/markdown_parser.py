from typing import List, Dict
import markdown_it

class MarkdownParser:
    """
    A robust Markdown parser for ingestion and chunking.
    """

    def __init__(self):
        self.md = markdown_it.MarkdownIt()

    def parse(self, file_path: str) -> List[Dict[str, str]]:
        """
        Parses a Markdown file and splits it into chunks based on headers.

        Args:
            file_path (str): The path to the Markdown file.

        Returns:
            List[Dict[str, str]]: A list of chunks, where each chunk is a dictionary
                                   containing the header and the content.
        """
        with open(file_path, 'r') as f:
            content = f.read()

        tokens = self.md.parse(content)
        chunks = []
        current_chunk = {"header": "", "content": ""}

        for i, token in enumerate(tokens):
            if token.type == 'heading_open':
                if current_chunk["content"]:
                    chunks.append(current_chunk)
                current_chunk = {"header": tokens[i+1].content, "content": ""}
            elif token.type == 'inline':
                if current_chunk["header"]:
                    current_chunk["content"] += token.content + '\n'
        
        if current_chunk["content"]:
            chunks.append(current_chunk)

        return chunks

if __name__ == '__main__':
    # Example usage
    parser = MarkdownParser()
    # Create a dummy markdown file for testing
    with open('test.md', 'w') as f:
        f.write("# Header 1\nThis is the first paragraph.\n## Header 2\nThis is the second paragraph.\n")
    
    chunks = parser.parse('test.md')
    for chunk in chunks:
        print(chunk)
