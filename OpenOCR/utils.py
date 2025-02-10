import io
import json
from math import cos
import os
from collections import defaultdict
from pydoc import text
from typing import Dict, List, Tuple, Union, Any

import cv2
import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from rapidfuzz.distance import DamerauLevenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_dict_to_file(data: Dict[str, Any], file_path: str) -> None:
    """Save a nested dictionary to a file, excluding image data."""
    processed_data = {
        dataset_name: {
            stroke_name: [label for label, _ in stroke_list]
            for stroke_name, stroke_list in dataset.items()
        }
        for dataset_name, dataset in data.items()
    }
    
    with open(file_path, 'w', encoding="utf8") as file:
        json.dump(processed_data, file, indent=4)

def read_vietnamese_characters(file_path: str) -> Dict[str, int]:
    """Read Vietnamese character mapping file."""
    char_to_num = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            char, num = line.strip().split()
            char_to_num[char] = int(num)
    return char_to_num

def list_to_string(numbers: List[int]) -> str:
    """Convert list of numbers to hyphen-separated string."""
    return '-'.join(map(str, numbers))

def read_stroke_mapping(file_path: str) -> Dict[str, str]:
    """Read stroke to word mapping file."""
    strokes_to_word = {}
    with open(file_path, 'r', encoding='utf8') as file:
        for line in file:
            word, strokes = line.strip().split()
            strokes_to_word[strokes] = word
    return strokes_to_word

def encode_vietnamese_character(char: str, char_to_num: Dict[str, int]) -> Union[int, str]:
    """Encode single character to number."""
    return char_to_num.get(char, "_")

def decode_vietnamese_character(num: int, char_to_num: Dict[str, int]) -> str:
    """Decode number to character."""
    for char, value in char_to_num.items():
        if value == num:
            return char
    return '_'

def encode_word(word: str, char_to_num: Dict[str, int]) -> List[Union[int, str]]:
    """Encode word to list of numbers."""
    return [encode_vietnamese_character(char, char_to_num) for char in word]

char_to_num = read_vietnamese_characters("./tools/utils/dict/strokes/Vietnamese-Characters-Simplified.txt")

def decode_word(strokes: str, char_to_num: Dict[str, int], 
                strokes_to_word: Dict[str, str]) -> str:
    """Decode strokes string to word using dictionary or character mapping."""
    if strokes in strokes_to_word:
        return strokes_to_word[strokes]
    return ''.join([decode_vietnamese_character(num, char_to_num) for num in map(int, strokes.split('-'))])

def read_confused_dict(filepath: str) -> Dict[str, List[str]]:
    """Read dictionary of confused words."""
    confused_word_dict = defaultdict(list)
    with open(filepath, "r", encoding="utf8") as file:
        for line in file:
            word, strokes = line.strip().split()
            confused_word_dict[strokes].append(word)
    return dict(confused_word_dict)

def rectify(stroke: str, list_strokes: List[str]) -> str:
    """Find most similar stroke pattern."""

    # Levenshtein similarity
    # Levenshtein_sim = [DamerauLevenshtein.normalized_similarity(stroke, i) for i in list_strokes]
    # return list_strokes[np.argmax(similarities)]

    # Cosine similarity
    all_strokes = [stroke] + list_strokes
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_strokes)
    
    # Compute cosine similarities (row 0 is the stroke)
    cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Total similarity
    # total_sim = [0.3*Levenshtein_sim[i] + 0.7*cos_sim[i] for i in range(len(list_strokes))]
    return list_strokes[np.argmax(cos_sim)]

def extract_features(encoder: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Extract features from image using encoder."""
    encoder.eval()
    # return encoder(image, text=None, return_loss=False, test_speed=False)
    return encoder(image, text=None, return_loss=False)

def get_support_data(
    folder_path: str = "/mlcv2/WorkingSpace/Personal/hamh/Ha/Data/SupportSamples_lmdb/",
    selected_datasets: List[str] = None
) -> Dict[str, Dict[str, List[Tuple[str, Image.Image]]]]:
    """Load and organize LMDB datasets."""
    if selected_datasets is None:
        selected_datasets = ['DaiTuZin', 'ThienAn', 'ThuongMai', 'TieuTuFull', 'Tieutuzin']
    
    data = {}
    for subdir in selected_datasets:
        lmdb_path = os.path.join(folder_path, subdir)
        if not os.path.isdir(lmdb_path):
            print(f"Dataset {subdir} not found. Skipping.")
            continue
            
        dataset_data = defaultdict(list)
        env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        
        try:
            with env.begin(write=False) as txn:
                n_samples = int(txn.get('num-samples'.encode()))
                for i in range(1, n_samples + 1):
                    image_bin = txn.get(f'image-{i:09d}'.encode())
                    label = txn.get(f'label-{i:09d}'.encode()).decode()
                    
                    if not (image_bin and label):
                        continue
                        
                    image = Image.open(io.BytesIO(image_bin)).convert('RGB')
                    strokes = '-'.join(map(str, encode_word(label, char_to_num)))
                    dataset_data[strokes].append((label, image))
                    
            print(f"Loaded dataset: {lmdb_path}")
        except Exception as e:
            print(f"Error loading {subdir}: {e}")
        finally:
            env.close()
            
        data[subdir] = dict(dataset_data)
    
    print(f"Loaded {len(data)} datasets")
    return data

def get_support_samples(
    strokes: str, 
    support_data: Dict[str, Dict[str, List[Tuple[str, Image.Image]]]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Get support samples for given strokes."""
    support_images = []
    support_labels = []
    for font_data in support_data.values():
        if strokes in font_data:
            labels, images = zip(*font_data[strokes])
            support_labels.extend(labels)
            support_images.extend(images)
    return np.array(support_images), np.array(support_labels)

def compare_image_features(image_features: torch.Tensor, support_features: torch.Tensor) -> torch.Tensor:
    """Compare two image feature matrices using cosine similarity."""
    return torch.nn.functional.cosine_similarity(image_features, support_features)

def resize_image(img: np.ndarray, img_w: int = 128, img_h: int = 32) -> np.ndarray:
    """Resize image to specified dimensions."""
    return cv2.resize(img, (img_w, img_h))

def handle_confusion(
    image: torch.Tensor,
    strokes: str,
    support_data: Dict[str, Dict[str, List[Tuple[str, Image.Image]]]],
    encoder: torch.nn.Module
) -> Dict[str, Any]:
    """Handle confused cases using support samples."""
    # Get support images and labels
    support_images, support_labels = get_support_samples(strokes, support_data)
    
    # Convert support images to tensors and preprocess
    support_images = torch.tensor(np.array([
        resize_image(np.array(img)) for img in support_images
    ])).permute(0, 3, 1, 2).float().to(device)
    
    # Ensure the input image is a 4D tensor
    if image.dim() == 3:
        image = image.unsqueeze(0)

    # Create a batch containing the input image and support images
    batch_images = torch.cat([image, support_images], dim=0) # [num_images, channels, height, width]
    
    # Extract features for the batch
    features = extract_features(encoder, batch_images) # [num_images, C, H, W] or similar
    
    # Flatten the features
    features = features.view(features.size(0), -1) # [num_images, feature_dim]
    
    # Separate features of the input image and support images
    image_features = features[0].unsqueeze(0)       # Shape: [1, feature_dim]
    support_features = features[1:]                 # Shape: [num_support, feature_dim]
    
    # Normalize the features
    image_features = F.normalize(image_features, p=2, dim=1)
    support_features = F.normalize(support_features, p=2, dim=1)
    
    # Compute cosine similarity between input image and support images
    cos_sim = torch.mm(image_features, support_features.t()).squeeze(0)  # Shape: [num_support]
    
    # Convert similarities to numpy array and find the index of the maximum similarity
    cos_sim = cos_sim.cpu().numpy()
    max_index = cos_sim.argmax()
    
    return {
        'closest_label': support_labels[max_index],
        'similarity': cos_sim.tolist()
    }