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
    text = re.sub(r'\[.*?\]:', '', text)
    return text.strip()

def get_word_count(text: str) -> int:
    return len(text.split())

def call_mistral_api(prompt: str, max_length: int = 200, temperature: float = 0.6, max_retries: int = 3, retry_delay: int = 2) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": max_length,
            "temperature": temperature,
            "top_p": 0.95,
            "return_full_text": False,
            "max_new_tokens": max_length
        }
    }

    api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    headers = {"Authorization": "Bearer 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"}

    for attempt in range(max_retries):
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            response_data = response.json()
            
            if isinstance(response_data, list) and len(response_data) > 0:
                generated_text = response_data[0].get('generated_text', '')
            else:
                generated_text = response_data.get('generated_text', '')

            if not generated_text:
                raise APIError("Empty response from API")
                
            return clean_generated_text(generated_text)

        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise APIError(f"Failed to communicate with API after {max_retries} attempts: {str(e)}")
            logger.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
            sleep(retry_delay)
        
        except (KeyError, IndexError) as e:
            raise APIError(f"Unexpected API response format: {str(e)}")

def generate_initial_justification(
    abstract: str,
    conclusion: str,
    keywords: List[str],
    conference_name: str,
) -> str:
    system_prompt = (
        "You are a precise AI that generates detailed justifications. "
        "Generate a justification of approximately 100 words explaining "
        "why a research paper fits a specific conference. Include specific details "
        "from the abstract and conclusion."
    )
    
    user_prompt = (
        f"Abstract: {abstract}\n\n"
        f"Conclusion: {conclusion}\n\n"
        f"Keywords: {', '.join(keywords)}\n\n"
        f"Conference: {conference_name}\n\n"
        "Generate a detailed justification of around 100 words."
    )

    return call_mistral_api(f"{system_prompt}\n\n{user_prompt}")

def generate_final_justification(initial_justification: str) -> str:
    system_prompt = (
        "You are a precise AI that creates concise summaries. "
        "Summarize the following justification in EXACTLY 50-70 words while "
        "maintaining the key points and specific details."
    )
    
    user_prompt = (
        f"Original justification:\n{initial_justification}\n\n"
        "Create a concise version between 50-70 words."
    )

    return call_mistral_api(f"{system_prompt}\n\n{user_prompt}", max_length=150)

def Doraemon_justification(
    abstract: str,
    conclusion: str,
    keywords: List[str],
    conference_name: str,
) -> Optional[str]:
    try:
        validate_inputs(abstract, conclusion, keywords, conference_name)
        
        # Step 1: Generate initial detailed justification
        logger.info("Generating initial detailed justification...")
        initial_justification = generate_initial_justification(
            abstract, conclusion, keywords, conference_name
        )
        initial_word_count = get_word_count(initial_justification)
        logger.info(f"Initial justification generated: {initial_word_count} words")
        
        # Step 2: Generate final concise justification
        logger.info("Generating final concise justification...")
        final_justification = generate_final_justification(initial_justification)
        final_word_count = get_word_count(final_justification)
        logger.info(f"Final justification generated: {final_word_count} words")
        
        # Verify final length
        if final_word_count < 50 or final_word_count > 70:
            logger.warning(f"Final justification length ({final_word_count} words) outside target range")
        
        return final_justification

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