import numpy as np
import os
import sys
import argparse
import torch
from PIL import Image
from Dino.model.dino_vision import DINO_Finetune
from Dino.utils.utils import Config
from Dino.metric.eval_acc import TextAccuracy
import time
import torchvision.transforms.functional as TF
from torchvision import transforms
import warnings

# Tắt cảnh báo UserWarning
warnings.filterwarnings("ignore", category=UserWarning)
totensor = transforms.ToTensor()
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
# Định nghĩa device (GPU hoặc CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image for inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint')
    args = parser.parse_args()
    return args

def load_model(config, checkpoint_path):
    model = DINO_Finetune(config)
    model = torch.nn.DataParallel(model).to(device)
    if checkpoint_path:
        # print(f"Loading model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['net'])
    model.eval()
    return model


def preprocess_image(image_path, config):
    # Load và resize ảnh
    image = Image.open(image_path).convert('RGB').resize((128, 32))
    
    # Chuyển ảnh thành tensor
    image = totensor(image)  # Chuyển từ PIL Image sang Tensor (đã chuẩn hóa về [0, 1])

    # Chuẩn hóa ảnh với mean và std của ImageNet
    image = TF.normalize(image, mean, std)

    # Đưa tensor về device (CPU hoặc GPU)
    image = image.to(device)
    # # print(image.shape)
    return image

def infer(model, image_tensor):
    with torch.no_grad():
        # print(image_tensor.unsqueeze(0))
        output = model(image_tensor.unsqueeze(0), text=None, return_loss=False, test_speed=False)
    return output

def post_process(output, label_convertor):
    label_indexes, label_scores = label_convertor.tensor2idx(output)
    pred_text = label_convertor.idx2str(label_indexes)
    return pred_text

def save_results_to_txt(image_path, result, algorithm_name):
    # Create directory for results if it doesn't exist
    result_dir = '/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/one_image/results'
    os.makedirs(result_dir, exist_ok=True)

    # Define file path with algorithm name
    file_path = os.path.join(result_dir, f"{algorithm_name}.txt")

    # Extract the image name (without path)
    image_name = os.path.basename(image_path)

    # Save the result in the format 'image_name infer \t result'
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"{image_name} \t {result}\n")
def main():
    args = _parse_arguments()

    # Load config and model
    config = Config(args.config)
    model = load_model(config, args.checkpoint)

    # Preprocess image
    image_tensor = preprocess_image(args.image_path, config)

    # Perform inference
    output = infer(model, image_tensor)

    # Post-process and get text from output
    result = post_process(output, model.module.label_convertor)[0]

    # # print the result
    print(f"Predicted text: {result}")
    save_results_to_txt(args.image_path, result, "CCD")
if __name__ == "__main__":
    main()