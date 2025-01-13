import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from Mistral7b_Instruct_2 import Doraemon_justification
from Binary_classification import DoraemonBinaryClassifier
from Conference_classification import DoraemonConferenceClassifier

def load_model(model, checkpoint_path, device):
    """Helper function to load model with correct state dict structure"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model
def process_saved_data(input_dir: Path, output_dir: Path):
    print("[INFO] Initializing processing of saved data...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    text_dir = input_dir / "texts"
    vector_dir = input_dir / "vectors"
    keywords_dir = input_dir / "keywords"

    for dir_path in [text_dir, vector_dir, keywords_dir]:
        if not dir_path.exists():
            raise ValueError(f"Directory not found: {dir_path}")
        
    vector_files = list(vector_dir.glob("*.pt"))
    if not vector_files:
        raise ValueError(f"No vector files found in {vector_dir}")
    print(f"[INFO] Found {len(vector_files)} files to process")

    sample_vector = torch.load(vector_files[0], map_location=device)
    input_dim = sample_vector.shape[0]
    print(f"[INFO] Detected input dimension: {input_dim}")

    try:
        binary_classifier = DoraemonBinaryClassifier(input_dim=input_dim).to(device)
        conference_classifier = DoraemonConferenceClassifier(input_dim=input_dim, num_classes=5).to(device)
        
        binary_classifier = load_model(binary_classifier, "doraemon_binary_classifier.pt", device)
        conference_classifier = load_model(conference_classifier, "doraemon_conference_classifier.pt", device)
        
        binary_classifier.eval()
        conference_classifier.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading models: {str(e)}")

    label_map = {0: "CVPR", 1: "TMLR", 2: "KDD", 3: "NEURIPS", 4: "EMNLP"}
    
    print("[INFO] Loading and processing saved data...")
    features_list = []
    file_ids = []
    
    for vector_file in tqdm(vector_files, desc="Loading vectors"):
        try:
            features = torch.load(vector_file, map_location=device)
            features_list.append(features)
            file_ids.append(vector_file.stem)
        except Exception as e:
            print(f"[WARNING] Error loading vector {vector_file}: {str(e)}")
            continue

    print("[INFO] Computing normalization statistics...")
    all_features = torch.stack(features_list)
    feature_mean = all_features.mean(dim=0)
    feature_std = all_features.std(dim=0) + 1e-6

    print("[INFO] Processing with normalized features...")
    results = []
    
    for idx, file_id in enumerate(tqdm(file_ids, desc="Processing files")):
        try:
            text_file = text_dir / f"{file_id}.json"
            keywords_file = keywords_dir / f"{file_id}.txt"
            
            with open(text_file, 'r') as f:
                parsed_content = json.load(f)
            
            with open(keywords_file, 'r') as f:
                keywords = f.read().splitlines()

            abstract = ""
            conclusion = ""
            for heading, content in parsed_content.items():
                if 'abstract' in heading.lower() or 'introduction' in heading.lower():
                    abstract = content
                elif 'conclusion' in heading.lower() or 'summary' in heading.lower():
                    conclusion = content

            if abstract == "" or conclusion == "":
                for heading, content in parsed_content.items():
                    if 'abstract' in content.lower() or 'introduction' in content.lower():
                        abstract = content
                        break
                for heading, content in parsed_content.items():
                    if 'conclusion' in content.lower() or 'summary' in content.lower():
                        conclusion = content
                        break

            normalized_features = (features_list[idx] - feature_mean) / feature_std
            
            with torch.no_grad():
                binary_pred = binary_classifier(normalized_features.unsqueeze(0).to(device))
                is_publishable = binary_pred.item() > 0.5
                
                conference = "na"
                justification = "na"
                
                if is_publishable:
                    conf_pred = conference_classifier(normalized_features.unsqueeze(0).to(device))
                    conference_id = torch.argmax(conf_pred).item()
                    conference = label_map[conference_id]
                    
                    justification = Doraemon_justification(
                        abstract=abstract,
                        conclusion=conclusion,
                        keywords=keywords,
                        conference_name=conference
                    )
            
            results.append([file_id, int(is_publishable), conference, justification])
            
        except Exception as e:
            print(f"[WARNING] Error processing results for {file_id}: {str(e)}")
            results.append([file_id, 0, 'error', f'Error: {str(e)}'])

    df = pd.DataFrame(results, columns=['Paper ID', 'Publishable', 'Conference', 'Rationale'])
    df.to_csv(output_dir / "results.csv", index=False)
    print(f"[INFO] Results saved to {output_dir / 'results.csv'}")

if __name__ == "__main__":
    input_dir = Path("Sample")
    output_dir = Path("Sample")
    process_saved_data(input_dir, output_dir)