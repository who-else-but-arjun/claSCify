import json
import random
import requests
from pathlib import Path

class TextCorruptor:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
        self.headers = {"Authorization": "Bearer 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"}
        self._test_api_connection()
        self.fallback_phrases = [
            "potato dreams fly upward", "singing mountains eat clouds",
            "blue ideas sleep furiously", "yesterday tomorrow today simultaneously"
        ]

    def _test_api_connection(self):
        try:
            response = requests.post(self.api_url, headers=self.headers, json={"inputs": "test"})
            response.raise_for_status()
        except Exception as e:
            print(f"API connection failed: {e}")

    def clean_text(self, text):
        if not isinstance(text, str):
            return text
        return ' '.join(text.replace('\n', ' ').split())

    def remove_characters(self, text):
        if not text:
            return text
        chars = list(text)
        remove_count = random.randint(len(chars) // 20, len(chars) // 10)
        for _ in range(remove_count):
            if chars:
                idx = random.randint(0, len(chars) - 1)
                chars.pop(idx)
        return ''.join(chars)

    def remove_words(self, text):
        words = text.split()
        if len(words) <= 1:
            return text
            
        remove_count = random.randint(len(words) // 6, len(words) // 4)
        for _ in range(remove_count):
            if words:
                idx = random.randint(0, len(words) - 1)
                words.pop(idx)
        return ' '.join(words)

    def remove_sentences(self, section):
        sentences = [s.strip() for s in section.split('.') if s.strip()]
        if len(sentences) <= 1:
            return section
            
        remove_count = random.randint(max(1, int(len(sentences) * 0.2)), 
                                    max(1, int(len(sentences) * 0.4)))
        for _ in range(remove_count):
            if sentences:
                idx = random.randint(0, len(sentences) - 1)
                sentences.pop(idx)
        return '. '.join(sentences) + '.' if sentences else ''

    def remove_paragraphs(self, text):
        paragraphs = text.split('. ')
        if len(paragraphs) <= 1:
            return text
            
        remove_count = random.randint(len(paragraphs) // 4, len(paragraphs) // 2)
        for _ in range(remove_count):
            if paragraphs:
                idx = random.randint(0, len(paragraphs) - 1)
                paragraphs.pop(idx)
        return '. '.join(paragraphs)

    def generate_nonsense(self):
        try:
            payload = {
                "inputs": "Generate a nonsensical phrase. it should be completely random and should be atleast 5 - 20 words",
                "parameters": {"max_length": 50, "temperature": 0.9}
            }
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            text = response.json()[0]["generated_text"].split(":")[-1].strip('"\'').strip()
            return text if text and len(text.split()) <= 5 else random.choice(self.fallback_phrases)
        except:
            return random.choice(self.fallback_phrases)

    def add_nonsense(self, section):
        words = section.split()
        if not words:
            return section
        num_phrases = random.randint(1, 2)
        for _ in range(num_phrases):
            if words:
                pos = random.randint(0, len(words))
                words.insert(pos, self.generate_nonsense())
        return ' '.join(words)

    def disturb_grammar(self, text):
        if not isinstance(text, str) or not text.strip():
            return text
        words = text.split()
        if len(words) < 2:
            return text
            
        for i in range(len(words)):
            if random.random() > 0.8:
                if words[i].lower() in {'a', 'an', 'the'}:
                    words[i] = ''
                elif len(words[i]) > 3:
                    if words[i].endswith('ing'):
                        words[i] = words[i][:-3] + 'ed'
                    elif words[i].endswith('ed'):
                        words[i] = words[i][:-2] + 'ing'
        return ' '.join(w for w in words if w)

    def reorder_text(self, text):
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) <= 1:
            return text
        random.shuffle(sentences)
        return '. '.join(sentences) + '.'

    def corrupt_document(self, data):
        cleaned_data = {self.clean_text(k): self.clean_text(v) for k, v in data.items()}
        
        # More moderate section removal (20-40% of sections)
        sections = list(cleaned_data.keys())
        if len(sections) > 1:
            remove_count = random.randint(
                max(1, int(len(sections) * 0.2)),
                max(1, int(len(sections) * 0.4))
            )
            for _ in range(remove_count):
                if sections:
                    cleaned_data.pop(random.choice(sections))
                    sections = list(cleaned_data.keys())
        
        corrupted = {}
        for heading, content in cleaned_data.items():
            # Apply removal operations with moderate probabilities
            if random.random() < 0.5:
                content = self.remove_paragraphs(content)
            if random.random() < 0.7:
                content = self.remove_sentences(content)
            if random.random() < 0.7:
                content = self.remove_words(content)
            if random.random() < 0.6:
                content = self.remove_characters(content)
            
            # Apply other corruptions
            if random.random() < 0.5:
                content = self.add_nonsense(content)
            if random.random() < 0.6:
                content = self.disturb_grammar(content)
            if random.random() < 0.5:
                content = self.reorder_text(content)
            
            # 30% chance to corrupt heading
            if random.random() < 0.4:
                heading = self.remove_words(heading)
                if random.random() < 0.4:
                    heading = self.remove_characters(heading)
            
            if content.strip():
                corrupted[heading] = content
        
        items = list(corrupted.items())
        random.shuffle(items)
        return dict(items)

def process_directory(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    corruptor = TextCorruptor()
    
    for json_file in input_path.glob('**/*.json'):
        output_file = output_path / json_file.name
        
        if output_file.exists():
            print(f"Skipping {json_file.name} - already processed")
            continue
            
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            corrupted_data = corruptor.corrupt_document(data)
            with open(output_file, 'w') as f:
                json.dump(corrupted_data, f, indent=4)
            print(f"Processed: {json_file.name} -> {output_file.name}")
        except Exception as e:
            print(f"Error processing {json_file}: {e}")

if __name__ == "__main__":
    input_dir = "Dataset/texts/publishable"
    output_dir = "Dataset/texts/non-publishable"
    process_directory(input_dir, output_dir)