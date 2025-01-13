import requests
import re
from typing import List, Optional
import logging
from time import sleep
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class APIError(Exception):
    pass

def validate_inputs(abstract: str, conclusion: str, keywords: List[str], conference_name: str) -> bool:
    if not isinstance(abstract, str):
        raise ValueError("Abstract must be a non-empty string")
    if not isinstance(conclusion, str):
        raise ValueError("Conclusion must be a non-empty string")
    if not isinstance(keywords, list) or not all(isinstance(k, str) for k in keywords):
        raise ValueError("Keywords must be a non-empty list of strings")
    if not isinstance(conference_name, str):
        raise ValueError("Conference name must be a non-empty string")
    return True

def clean_generated_text(text: str) -> str:
    text = ' '.join(text.split())
    text = re.sub(r'<\|.*?\|>', '', text)
    # Add additional cleaning for common Mistral formatting
    text = re.sub(r'\[.*?\]:', '', text)  # Remove prefixes like [Assistant]:
    return text.strip()

def get_word_count(text: str) -> int:
    return len(text.split())

def Doraemon_justification(
    abstract: str,
    conclusion: str,
    keywords: List[str],
    conference_name: str,
    max_retries: int = 3,
    retry_delay: int = 2
) -> Optional[str]:
    try:
        validate_inputs(abstract, conclusion, keywords, conference_name)
        
        # Modified system prompt to be more explicit about output format
        system_prompt = (
            "You are a precise AI that generates very brief justifications. "
            "Generate a justification of EXACTLY 75 words or less explaining "
            "why a research paper fits a specific conference. Be extremely concise."
        )
        
        user_prompt = (
            f"Abstract: {abstract}\n\n"
            f"Keywords: {', '.join(keywords)}\n\n"
            f"Conference: {conference_name}\n\n"
            "Important: Keep your response under 75 words and focus only on the most relevant points."
        )

        # logger.info(f"Input word counts:")
        # logger.info(f"Abstract: {get_word_count(abstract)} words")
        # logger.info(f"Conclusion: {get_word_count(conclusion)} words")
        # logger.info(f"Keywords: {len(keywords)} keywords")

        payload = {
            "inputs": f"{system_prompt}\n\n{user_prompt}",
            "parameters": {
                "max_length": 150,
                "temperature": 0.6, 
                "top_p": 0.95,
                "return_full_text": False,
                "max_new_tokens": 150  
            }
        }

        api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
        headers = {"Authorization": "Bearer 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"}

        for attempt in range(max_retries):
            try:
                response = requests.post(api_url, headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                response_data = response.json()
                
                # Improved response handling
                if isinstance(response_data, list) and len(response_data) > 0:
                    generated_text = response_data[0].get('generated_text', '')
                else:
                    generated_text = response_data.get('generated_text', '')

                if not generated_text:
                    raise APIError("Empty response from API")
                    
                generated_text = clean_generated_text(generated_text)
                word_count = get_word_count(generated_text)
                logger.info(f"Generated text word count: {word_count} words")
                
                # Verify minimum length to ensure completeness
                if word_count < 50:  # Minimum word count threshold
                    logger.warning(f"Generated text appears incomplete: only {word_count} words")
                    raise APIError("Generated text appears incomplete")

                return generated_text

            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise APIError(f"Failed to communicate with API after {max_retries} attempts: {str(e)}")
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                sleep(retry_delay)
            
            except (KeyError, IndexError) as e:
                raise APIError(f"Unexpected API response format: {str(e)}")

    except Exception as e:
        logger.error(f"Error generating justification: {str(e)}")
        raise

def main(abstract, conclusion, keywords, conference_name):
    try:
        justification = Doraemon_justification(abstract, conclusion, keywords, conference_name)
        
        if justification:
            print("\n[INFO] Generated Justification:")
            print("-" * 80)
            print(justification)
            print("-" * 80)
            print(f"Word count: {get_word_count(justification)} words")
        return justification
    except (ValueError, APIError) as e:
        logger.error(f"Failed to generate justification: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    abstract = "This paper presents a novel approach to optimizing neural network architectures using evolutionary algorithms."
    conclusion = "The results demonstrate significant improvements in accuracy and efficiency, making this method suitable for deployment in real-world AI systems."
    keywords = ["neural networks", "evolutionary algorithms", "optimization", "AI"]
    conference_name = "NeurIPS 2025"
    main(abstract, conclusion, keywords, conference_name)