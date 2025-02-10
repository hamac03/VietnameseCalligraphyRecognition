import torch.nn as nn
import torch.nn.functional as F
import torch
import editdistance as ed
from tqdm import tqdm
import re
import time
import os
from PIL import Image
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextAccuracy(nn.Module):
    def __init__(self, charset_path, case_sensitive, model_eval):
        self.charset_path = charset_path
        self.case_sensitive = case_sensitive
        # print(f'Case_sensitive in eval: {case_sensitive}')
        self.model_eval = model_eval
        assert self.model_eval in ['vision', 'language', 'alignment']
        self._names = ['ccr', 'cwr', 'ted', 'ned', 'ted/w', 'words', 'time']

        self.total_num_char = 0.
        self.total_num_word = 0.
        self.correct_num_char = 0.
        self.correct_num_word = 0.
        self.total_ed = 0.
        self.total_ned = 0.
        self.inference_time = 0.

        self.fail_predict_dir = 'fail_predict/file_images'
        os.makedirs(self.fail_predict_dir, exist_ok=True)

    def compute(self, model, dataloader):
        test_data_loader_iter = iter(dataloader)
        # count = 0
        print(len(test_data_loader_iter))
        for test_iter in tqdm(range(len(test_data_loader_iter))):
            image_tensors, label_tensors = test_data_loader_iter.next()
            # print(image_tensors.shape)
            # print(image_tensors.min())
            # print(image_tensors.max())

            image_tensors = image_tensors.to(device)
            start_time = time.time()
            out_dec = model(image_tensors, text=None, return_loss=False, test_speed=False)
            label_indexes, label_scores = model.module.label_convertor.tensor2idx(out_dec)
            pt_text = model.module.label_convertor.idx2str(label_indexes)
            self.inference_time += time.time() - start_time
            gt_text = label_tensors[0]
            # print(gt_text)
            # comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')
            comp = re.compile(r"[^!\"#$%&'()*+,\-./0123456789:;<=>?@A-Z\[\\\]_`a-z~ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ]")
            for i in range(len(gt_text)):
                # print(f"Original Ground Truth: {gt_text[i]}")
                # print(f"Original Prediction: {pt_text[i]}")

                if not self.case_sensitive:
                    # Nếu không phân biệt chữ hoa và chữ thường
                    gt_text_lower = gt_text[i].lower()
                    pred_text_lower = pt_text[i].lower()
                    gt_text_lower_ignore = comp.sub('', gt_text_lower)
                    pred_text_lower_ignore = comp.sub('', pred_text_lower)
                    # In ra giá trị sau khi đã chuyển sang chữ thường và lọc các ký tự không hợp lệ
                    print(f"Filtered Ground Truth: {gt_text_lower_ignore}")
                    print(f"Filtered Prediction: {pred_text_lower_ignore}")
                else:
                    # Nếu có phân biệt chữ hoa và chữ thường
                    gt_text_lower_ignore = comp.sub('', gt_text[i])
                    pred_text_lower_ignore = comp.sub('', pt_text[i])
                    # In ra giá trị sau khi đã lọc các ký tự không hợp lệ
                    print(f"Filtered Ground Truth (case sensitive): {gt_text_lower_ignore}")
                    print(f"Filtered Prediction (case sensitive): {pred_text_lower_ignore}\n")

                if gt_text_lower_ignore == pred_text_lower_ignore:
                    self.correct_num_word += 1

                """This comment use for saving fail images"""
                # else:
                #     # print(gt_text[i], pt_text[i], image_tensors[i])
                #     self._save_failed_prediction(test_iter, i, gt_text[i], pt_text[i], image_tensors[i])
                distance = ed.eval(gt_text_lower_ignore, pred_text_lower_ignore)
                self.total_ed += distance
                self.total_ned += float(distance) / max(len(gt_text[i]), 1)
                self.total_num_word += 1

                for j in range(min(len(gt_text[i]), len(pt_text[i]))):
                    if gt_text[i][j] == pt_text[i][j]:
                        self.correct_num_char += 1
                self.total_num_char += len(gt_text[i])

        mets = [self.correct_num_char / self.total_num_char,
                self.correct_num_word / self.total_num_word,
                self.total_ed,
                self.total_ned,
                self.total_ed / self.total_num_word,
                self.total_num_word,
                self.inference_time/len(test_data_loader_iter)]
        return dict(zip(self._names, mets))
    
    def _save_failed_prediction(self,batch_iter, index, gt_text, pt_text, image_tensor):
        # Create an image file name based on the index and predictions
        image_filename = f'image_{batch_iter}_{index}_{pt_text}_{gt_text}.jpg'
        image_path = os.path.join(self.fail_predict_dir, image_filename)

        # Denormalize the image tensor (assuming it's in the range [-1, 1])
        image_tensor = image_tensor.cpu()  # Move to CPU if it's on GPU
        image_tensor = (image_tensor + 1) / 2 * 255  # Scale to [0, 255]
        
        # Ensure the tensor is of type uint8
        image_tensor = image_tensor.clamp(0, 255).byte()

        # Convert the tensor to numpy and reshape if necessary
        image_array = image_tensor.permute(1, 2, 0).numpy()  # Change shape to (H, W, C)
        
        # Create a PIL image
        image = Image.fromarray(image_array)

        # Save the image
        image.save(image_path)

        # Save the label information
        label_filename = f'label_{index}.txt'
        with open(os.path.join(self.fail_predict_dir, label_filename), 'a') as f:
            f.write(f'{image_filename}\t{pt_text}\t{gt_text}\n')