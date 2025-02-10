import os

# Đường dẫn đến file groundtruth và các file dự đoán
groundtruth_file = "/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/STR_prediction_results/STR/groundtruth_test.txt"
prediction_files = {
    "SMTR": "/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/STR_prediction_results/STR/SMTR.txt",
    "CCD": "/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/STR_prediction_results/STR/CCD.txt",
    "VietOCR": "/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/STR_prediction_results/STR/VietOCR.txt",
    "PARSeq": "/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/STR_prediction_results/STR/ParseQ.txt",
    "ABINet": "/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/STR_prediction_results/STR/ABINet.txt",
    "SRN": "/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/STR_prediction_results/STR/SRN.txt",
    "ViTSTR": "/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/STR_prediction_results/STR/ViTSTR.txt"
}

# Đọc groundtruth
groundtruth = {}
with open(groundtruth_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            groundtruth[parts[0]] = parts[1]

# Đọc dự đoán từ các phương pháp
predictions = {method: {} for method in prediction_files}
for method, file_path in prediction_files.items():
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                predictions[method][parts[0]] = parts[1]

# Tạo danh sách các trường hợp giống và khác
same_predictions = []
different_predictions = []

for image_path, gt_label in groundtruth.items():
    pred_labels = {method: predictions[method].get(image_path, "N/A") for method in prediction_files}
    unique_predictions = set(pred_labels.values())
    
    if len(unique_predictions) == 1:  # Tất cả dự đoán giống nhau
        same_predictions.append((image_path, gt_label, pred_labels))
    else:  # Dự đoán khác nhau
        different_predictions.append((image_path, gt_label, pred_labels))
print(different_predictions)
# Hàm tạo HTML
def create_html(output_file, data, title):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("<html><head><title>Prediction Comparison</title></head><body>")
        f.write(f"<h1>{title}</h1>")
        f.write("<table border='1' style='border-collapse: collapse; text-align: center;'>")
        f.write("<tr><th>#</th><th>Image</th><th>Groundtruth</th>" + "".join([f"<th>{method}</th>" for method in prediction_files]) + "</tr>")
        
        for idx, (image_path, gt_label, pred_labels) in enumerate(data, 1):
            f.write("<tr>")
            f.write(f"<td>{idx}</td>")  # Số thứ tự
            f.write(f"<td><img src='{image_path}' width='100'></td>")  # Hiển thị ảnh trực tiếp từ đường dẫn
            f.write(f"<td>{gt_label}</td>")   # Thêm nhãn gốc
            
            # Thêm nhãn dự đoán từ các phương pháp
            for method in prediction_files:
                pred_label = pred_labels[method]
                color = "red" if pred_label != gt_label else "green"
                f.write(f"<td style='color: {color};'>{pred_label}</td>")
            
            f.write("</tr>")
        
        f.write("</table></body></html>")

# Tạo hai file HTML
output_same = "/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/STR_prediction_results/visulization/visualizationsame_predictions.html"
output_different = "/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/STR_prediction_results/visulization/visualizationdifferent_predictions.html"

create_html(output_same, same_predictions, "Same Predictions")
create_html(output_different, different_predictions, "Different Predictions")

print(f"Same predictions saved to {output_same}")
print(f"Different predictions saved to {output_different}")