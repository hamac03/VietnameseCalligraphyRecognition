import os
import sys
import numpy as np
import torch
from PIL import Image
import glob
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

from tools.engine import Config, Trainer
from tools.utility import ArgsParser

def parse_args():
    parser = ArgsParser()
    args = parser.parse_args()
    return args

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
    # Parse command-line arguments
    FLAGS = parse_args()

    # Load configuration
    cfg = Config(FLAGS.config)
    FLAGS = vars(FLAGS)
    opt = FLAGS.pop('opt', {})
    cfg.merge_dict(FLAGS)
    cfg.merge_dict(opt)

    # Initialize the trainer in inference mode
    trainer = Trainer(cfg, mode='infer', support_data=None)

    # Load and preprocess the input image
    image_path_ = FLAGS['image_path']
    file_paths = glob.glob(f"/mlcv2/WorkingSpace/Personal/hamh/Ha/Data/Vicalligraphy/ViCalligraphy/test_folder_images/images/*")
    with open('/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/STR_prediction_results/STR/SMTR.txt', encoding='utf-8', mode='a') as file:
        
        for image_path in file_paths:
            # Read image
            image = Image.open(image_path).convert('RGB').resize((64, 64))
            image = np.array(image)[:, :, ::-1].copy()  # Convert RGB to BGR
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1)  # Convert from [H, W, C] to [C, H, W]

            # Perform inference
            label = trainer.infer(image)[0]
            file.write(f"{image_path}\t{label}\n")
            # Print the result
            # print(f"Predicted label: {label}")
        # save_results_to_txt(image_path, label, "SMTR")
if __name__ == '__main__':
    main()
