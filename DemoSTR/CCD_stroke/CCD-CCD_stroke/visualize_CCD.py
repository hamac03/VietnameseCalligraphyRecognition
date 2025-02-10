import os

# Đường dẫn đến file groundtruth
groundtruth_file = "/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/STR_prediction_results/STR/groundtruth_test.txt"

# Đường dẫn các file dự đoán theo từng phương pháp và dataset
prediction_files = {}
methods = ["CCD", "CCD_Stroke"]
# datasets = ["ViCalligraphy", "ViCalligraphy_10000", "1m_ViCalligraphy", "1m_ViCalligraphy_10000"]
datasets = ["ViCalligraphy"]

for method in methods:
    for dataset in datasets:
        method_dataset = f"{method}_{dataset}"
        prediction_files[method_dataset] = f"/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/CCD_stroke/Results/{method}/{dataset}.txt"

# Đọc groundtruth
groundtruth = {}
with open(groundtruth_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) == 2:
            groundtruth[parts[0]] = parts[1]

# Đọc dự đoán từ các phương pháp và dataset
predictions = {method_dataset: {} for method_dataset in prediction_files}
for method_dataset, file_path in prediction_files.items():
    if os.path.exists(file_path):  # Kiểm tra file có tồn tại không
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    predictions[method_dataset][parts[0]] = parts[1]
    else:
        print(f"Warning: {file_path} does not exist!")

# Tạo danh sách các trường hợp giống và khác
same_predictions = []
different_predictions = []

for image_path, gt_label in groundtruth.items():
    gt_label_lower = gt_label.lower()  # Chuẩn hóa nhãn gốc thành chữ thường
    pred_labels = {
        method_dataset: predictions[method_dataset].get(image_path, "N/A")
        for method_dataset in prediction_files
    }
    pred_labels_lower = {key: value.lower() for key, value in pred_labels.items()}  # Chuẩn hóa dự đoán thành chữ thường
    unique_predictions = set(pred_labels_lower.values())
    
    if len(unique_predictions) == 1 and gt_label_lower in unique_predictions:  # Dự đoán giống nhãn gốc
        same_predictions.append((image_path, gt_label, pred_labels))
    else:  # Dự đoán khác nhau hoặc không trùng với nhãn gốc
        different_predictions.append((image_path, gt_label, pred_labels))

# Hàm tạo HTML
def create_html(output_file, data, title):
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("""
<html>
<head>
    <title>Prediction Comparison</title>
    <style>
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid black; padding: 5px; text-align: center; }
        th { background-color: #f2f2f2; }
        .correct { color: green; }
        .incorrect { color: red; }
    </style>
</head>
<body>
        """)
        f.write(f"<h1>{title}</h1>")
        f.write("<table>")
        f.write("<thead><tr><th>#</th><th>Image</th><th>Groundtruth</th>" + "".join([f"<th>{key}</th>" for key in prediction_files]) + "</tr></thead>")
        f.write("<tbody>")
        
        for idx, (image_path, gt_label, pred_labels) in enumerate(data, 1):
            f.write("<tr>")
            f.write(f"<td>{idx}</td>")  # Số thứ tự
            f.write(f"<td><img src='{image_path}' width='100'></td>")  # Hiển thị ảnh trực tiếp từ đường dẫn
            f.write(f"<td>{gt_label}</td>")   # Thêm nhãn gốc
            
            # Thêm nhãn dự đoán từ các phương pháp
            for key in prediction_files:
                pred_label = pred_labels[key]
                color_class = "correct" if pred_label == gt_label else "incorrect"
                f.write(f"<td class='{color_class}'>{pred_label}</td>")
            
            f.write("</tr>")
        
        f.write("</tbody></table></body></html>")

# Tạo hai file HTML
output_same = "/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/CCD_stroke/CCD-CCD_stroke/visualizationsame_predictions.html"
output_different = "/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/CCD_stroke/CCD-CCD_stroke/visualizationdifferent_predictions.html"

create_html(output_same, same_predictions, "Same Predictions")
create_html(output_different, different_predictions, "Different Predictions")

print(f"Same predictions saved to {output_same}")
print(f"Different predictions saved to {output_different}")
