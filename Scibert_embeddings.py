import json
from pathlib import Path
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textstat import textstat
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import softmax
import numpy as np
from typing import Dict, List, Tuple
import os
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import threading

class DoraemonProcessor:
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"[INFO] Model loaded on: {self.device}")
        self.hidden_size = 768
        self.lock = threading.Lock()

    def get_chunks(self, text: str, max_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = tokens["input_ids"][0]
        attention_mask = tokens["attention_mask"][0]
        num_tokens = input_ids.shape[0]
        chunk_boundaries = range(0, num_tokens, max_length)
        chunks_ids = []
        chunks_mask = []

        for start in chunk_boundaries:
            end = min(start + max_length, num_tokens)
            chunk_ids = input_ids[start:end]
            chunk_mask = attention_mask[start:end]
            if chunk_ids.shape[0] < max_length:
                pad_length = max_length - chunk_ids.shape[0]
                chunk_ids = torch.nn.functional.pad(chunk_ids, (0, pad_length), value=self.tokenizer.pad_token_id)
                chunk_mask = torch.nn.functional.pad(chunk_mask, (0, pad_length), value=0)
            chunks_ids.append(chunk_ids)
            chunks_mask.append(chunk_mask)

        all_chunk_ids = torch.stack(chunks_ids).unsqueeze(0)
        all_chunk_masks = torch.stack(chunks_mask).unsqueeze(0)
        return all_chunk_ids.to(self.device), all_chunk_masks.to(self.device)

    @torch.no_grad()
    def process_text(self, chunks_ids: torch.Tensor, chunks_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        all_embeddings = []
        all_hidden_states = []

        for i in range(chunks_ids.shape[1]):
            chunk_ids = chunks_ids[:, i, :]
            chunk_mask = chunks_mask[:, i, :]
            outputs = self.model(input_ids=chunk_ids, attention_mask=chunk_mask, output_hidden_states=True)
            all_embeddings.append(outputs.last_hidden_state)
            all_hidden_states.append(torch.stack(outputs.hidden_states[-4:]))

        final_embeddings = torch.cat(all_embeddings, dim=1)
        final_mask = chunks_mask.squeeze(0).reshape(-1)
        final_mask = final_mask[:final_embeddings.shape[1]]
        return final_embeddings, final_mask, torch.stack(all_hidden_states)

    def aggregate_embeddings(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        
        mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size())
        masked_embeddings = embeddings * mask_expanded
        
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        valid_tokens = torch.sum(attention_mask, dim=1, keepdim=True)
        valid_tokens = torch.clamp(valid_tokens, min=1e-9)
        mean_pooled = sum_embeddings / valid_tokens
        
        masked_embeddings_for_max = masked_embeddings.clone()
        masked_embeddings_for_max[~mask_expanded.bool()] = float('-inf')
        max_pooled = torch.max(masked_embeddings_for_max, dim=1)[0]
        
        attention_weights = torch.mean(embeddings, dim=-1)
        attention_weights = attention_weights.masked_fill(~attention_mask.bool(), float('-inf'))
        attention_scores = softmax(attention_weights, dim=1).unsqueeze(-1)
        attention_pooled = torch.sum(attention_scores * embeddings, dim=1)

        return {"mean": mean_pooled, "max": max_pooled, "attention": attention_pooled}

    def _calculate_statistical_features(self, text: str) -> Dict[str, float]:
        words = text.split()
        sentences = text.split('.')
        unique_words = len(set(words))
        total_words = max(1, len(words))
        total_sentences = max(1, len(sentences))

        return {
            "word_count": float(total_words),
            "sentence_count": float(total_sentences),
            "avg_word_length": sum(len(word) for word in words) / total_words,
            "avg_sentence_length": total_words / total_sentences,
            "lexical_diversity": unique_words / total_words
        }

    def _calculate_readability_scores(self, text: str) -> Dict[str, float]:
        return {
            "flesch_reading_ease": float(textstat.flesch_reading_ease(text)),
            "gunning_fog": float(textstat.gunning_fog(text)),
            "smog_index": float(textstat.smog_index(text)),
            "automated_readability_index": float(textstat.automated_readability_index(text)),
            "dale_chall_score": float(textstat.dale_chall_readability_score(text)),
            "difficult_words": float(textstat.difficult_words(text)),
            "linsear_write_formula": float(textstat.linsear_write_formula(text))
        }
    def create_weight_vectors(self, total_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        component_sizes = [768, 768, 768, 768, 3072, 5, 7, 20] 
        weight1 = torch.ones(total_size)
        weight2 = torch.ones(total_size)
        component_weights_v1 = [1.2, 1.2, 1.2, 1.5, 1.0, 1.0, 1.0, 0.8]
        component_weights_v2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
        start_idx = 0
        for i, size in enumerate(component_sizes):
            weight1[start_idx:start_idx + size] *= component_weights_v1[i]
            weight2[start_idx:start_idx + size] *= component_weights_v2[i]
            start_idx += size
        return weight1, weight2

    def _extract_topics(self, text: str, n_topics: int = 1, num_keywords: int = 20) -> Tuple[torch.Tensor, List[Tuple[str, float]]]:
        vectorizer = CountVectorizer(stop_words="english", min_df=1)
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        text_vectorized = vectorizer.fit_transform([text])
        lda.fit(text_vectorized)
        feature_names = vectorizer.get_feature_names_out()
        
        keywords = []
        for topic in lda.components_:
            top_indices = topic.argsort()[:-(num_keywords):][::-1]
            topic_keywords = [(feature_names[i], float(topic[i])) for i in top_indices]
            keywords.extend(topic_keywords)
        
        keywords = sorted(keywords, key=lambda x: x[1], reverse=True)[:num_keywords]
        
        keywords_embeddings = torch.zeros(num_keywords, device=self.device)
        for i, (keyword, weight) in enumerate(keywords):
            with torch.no_grad():
                keyword_tokens = self.tokenizer(
                    keyword, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                )
                input_ids = keyword_tokens["input_ids"].clone().to(self.device)
                attention_mask = keyword_tokens["attention_mask"].clone().to(self.device)
                
                keyword_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                keyword_embedding = keyword_outputs.last_hidden_state.clone().mean()
                keywords_embeddings[i] = keyword_embedding * weight
        
        keywords_embeddings = keywords_embeddings.detach().clone()
        return keywords_embeddings, keywords
    
    def process_document(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[str, float]]]:
        chunks_ids, chunks_mask = self.get_chunks(text)
        embeddings, attention_mask, hidden_states = self.process_text(chunks_ids, chunks_mask)
        
        pooled_embeddings = self.aggregate_embeddings(embeddings, attention_mask)
        mean_pooled = pooled_embeddings["mean"]
        max_pooled = pooled_embeddings["max"]
        attention_pooled = pooled_embeddings["attention"]
        cls_embeddings = embeddings[:, 0, :]
        layer_wise_embeddings = hidden_states[..., 0, :].mean(dim=0).unsqueeze(0)
        
        statistical_features = torch.tensor(list(self._calculate_statistical_features(text).values())).to(self.device)
        readability_scores = torch.tensor(list(self._calculate_readability_scores(text).values())).to(self.device)
        keyword_embeddings, keywords = self._extract_topics(text)
        
        combined_features = torch.cat([
            max_pooled.flatten(),
            mean_pooled.flatten(),
            attention_pooled.flatten(),
            cls_embeddings.flatten(),
            layer_wise_embeddings.flatten(),
            statistical_features,
            readability_scores,
            keyword_embeddings.flatten()
        ])
        print(f"[INFO] Keyword Embeddings Length : {keyword_embeddings.size()}")
        weight1, weight2 = self.create_weight_vectors(combined_features.size(0))
        return combined_features, weight1, weight2, keywords

    def process_single_file(self, file_data: tuple) -> Tuple[bool, torch.Tensor, torch.Tensor]:
        json_file, input_path, vector_output_path, keywords_output_path, first_vector_size = file_data
        try:
            relative_path = os.path.relpath(json_file.parent, input_path)
            vector_dir = vector_output_path / relative_path
            keywords_dir = keywords_output_path / relative_path
            
            with self.lock:
                vector_dir.mkdir(parents=True, exist_ok=True)
                keywords_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n[INFO] Processing file: {json_file}")

            with open(json_file, "r") as f:
                sections = json.load(f)

            text = ""
            for heading, content in sections.items():
                text += f"{heading}\n{content}\n\n"

            with self.lock:
                print(f"[INFO] Extracting features from text (length: {len(text)} chars)")
            
            combined_features, weight1, weight2, keywords = self.process_document(text)
            print(f"[INFO] Combined Features Length: {combined_features.size()}")

            current_size = combined_features.size(0)
            if first_vector_size is not None:
                assert current_size == first_vector_size, f"Vector size mismatch: {current_size} vs {first_vector_size}"

            vector_file = vector_dir / f"{json_file.stem}.pt"
            torch.save(combined_features, vector_file)
            keywords_file = keywords_dir / f"{json_file.stem}.txt"
            with open(keywords_file, 'w', encoding='utf-8') as f:
                for keyword, weight in keywords:
                    f.write(f"{keyword}\n")
            
            with self.lock:
                print(f"[SUCCESS] Saved vector features to: {vector_file}")
                print(f"[SUCCESS] Saved keywords to: {keywords_file}")
            
            return True, weight1, weight2

        except Exception as e:
            with self.lock:
                print(f"[ERROR] Failed to process {json_file}: {str(e)}")
            return False, None, None

    def process_json_files(self, input_dir: str, output_dir: str, max_workers: int = 4):
        print(f"[INFO] Starting processing from input directory: {input_dir}")
        print(f"[INFO] Output will be saved to: {output_dir}")
        print(f"[INFO] Using {max_workers} worker threads")

        processed_count = 0
        total_files = 0
        first_vector_size = None
        weights_saved = False
        all_files = []
        # for category in ["publishable", "non-publishable"]:
        #     input_path = Path(input_dir) / category
        #     vector_output_path = Path(output_dir) / "vectors" / category
        #     keywords_output_path = Path(output_dir) / "keywords" / category
            
        #     print(f"\n[INFO] Collecting files from category: {category}")
            
        #     for root, dirs, files in os.walk(input_path):
        #         json_files = [Path(root) / f for f in files if f.lower().endswith('.json')]
        #         all_files.extend([(f, input_path, vector_output_path, keywords_output_path, first_vector_size) for f in json_files])
        #         total_files += len(json_files)
        
        input_path = Path(input_dir) 
        vector_output_path = Path(output_dir) / "vectors"
        keywords_output_path = Path(output_dir) / "keywords" 
            
        for root, dirs, files in os.walk(input_path):
            json_files = [Path(root) / f for f in files if f.lower().endswith('.json')]
            all_files.extend([(f, input_path, vector_output_path, keywords_output_path, first_vector_size) for f in json_files])
            total_files += len(json_files)

        print(f"\n[INFO] Found {total_files} files to process")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for file_data in all_files:
                future = executor.submit(self.process_single_file, file_data)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                success, weight1, weight2 = future.result()
                if success:
                    processed_count += 1
                    if not weights_saved and weight1 is not None and weight2 is not None:
                        weight_path1 = Path(output_dir) / "weight1.pt"
                        weight_path2 = Path(output_dir) / "weight2.pt"
                        torch.save(weight1, weight_path1)
                        torch.save(weight2, weight_path2)
                        print(f"[SUCCESS] Saved weight vectors")
                        weights_saved = True

                    print(f"[PROGRESS] Processed {processed_count}/{total_files} files ({(processed_count/total_files)*100:.1f}%)")

        print(f"\n[COMPLETE] Processing finished. Total files processed: {processed_count}/{total_files}")
        if first_vector_size is not None:
            print(f"Vector size for all processed files: {first_vector_size}")

if __name__ == "__main__":
    input_dir = "Sample/texts"
    output_dir = "Sample"
    processor = DoraemonProcessor()
    processor.process_json_files(input_dir, output_dir, max_workers=1)