from PIL import Image
import argparse
import os

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
def predict_text_from_image(image_path, config):
    # Initialize the predictor with the config
    detector = Predictor(config)

    # Open the image using PIL
    img = Image.open(image_path)

    # Predict text from the image
    result = detector.predict(img)
    
    return result

def _parse_arguments():
    parser = argparse.ArgumentParser(description="Text Prediction from Image")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image for prediction")
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
    # Parse the command line arguments
    args = _parse_arguments()

    # Load the config
    config = Cfg.load_config_from_name('vgg_transformer')
    config['weights'] = './weights/transformerocr.pth'
    config['cnn']['pretrained'] = False
    config['device'] = 'cuda:0'

    # Predict text from the image
    result = predict_text_from_image(args.image_path, config)
    print(f"Predicted text: {result}")
    save_results_to_txt(args.image_path, result, "VietOCR")
    
if __name__ == "__main__":
    main()
