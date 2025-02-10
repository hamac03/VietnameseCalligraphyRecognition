import streamlit as st
import os
import shutil
import subprocess
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
from streamlit_drawable_canvas import st_canvas

def clear_results_folder(folder_path):
    try:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        print("Đã xóa tất cả các file trong thư mục.")
    except Exception as e:
        print(f"Lỗi khi xóa file: {e}")

st.title("NHẬN DẠNG CHỮ THƯ PHÁP TIẾNG VIỆT")

# Phần sidebar: Lựa chọn phương pháp và tải ảnh
with st.sidebar:
    st.header("Cài đặt")
    # Chọn GPU
    gpu = st.selectbox(
        "GPU",
        ["0", "1", "2", "3", "4", "5", "6", "7"]
    )

    # Chọn loại bài toán
    task_type = st.radio(
        "Hướng tiếp cận",
        ["STR APPROACH", "CCD APPROACH"]
    )
    
    # Chọn phương pháp theo loại bài toán
    if task_type == "STR APPROACH":
        method = st.radio(
            "Phương pháp nhận dạng",
            ["VietOCR", "CCD", "SMTR", "SRN", "ABINet", "ViTSTR", "SVTR", "PARSeq"]
        )
    elif task_type == "CCD APPROACH":
        method = st.radio(
            "Chọn phương pháp CCD:",
            ["CCD", "CCD-SLD"]
        )

        training_data = st.radio(
            "Dữ liệu training:",
            ["ViCalligraphy", 
            "ViCalligraphy_10000", 
            "1m_ViCalligraphy", 
            "1m_ViCalligraphy_10000"],
            format_func=lambda x: {
                "ViCalligraphy": "ViCalligraphy",
                "ViCalligraphy_10000": "ViCalligraphy + ViCalligraphySynth",
                "1m_ViCalligraphy": "ViSynth1m + ViCalligraphy",
                "1m_ViCalligraphy_10000": "ViSynth1m + ViCalligraphy + ViCalligraphySynth"
            }[x]
        )
    
    # Tải ảnh và nút chạy nhận dạng
    uploaded_file = st.file_uploader("Tải lên ảnh để nhận dạng", type=["jpg", "png", "jpeg"])

# Danh sách các phương pháp
methods_list = ["VietOCR", "CCD", "SMTR", "SRN", "ABINet", "ViTSTR", "SVTR", "PARSeq"]

if uploaded_file:
    # Mở ảnh từ file đã tải lên
    image = Image.open(uploaded_file)

    # Resize ảnh khi hiển thị (ví dụ: tăng chiều rộng lên 2 lần)
    base_width = 500  # Chiều rộng mà bạn muốn hiển thị
    w_percent = (base_width / float(image.width))
    h_size = int((float(image.height) * float(w_percent)))
    resized_image = image.resize((base_width, h_size))

    # Cài đặt canvas cho phép người dùng crop ảnh
    # st.subheader("Vẽ vùng cần crop")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",  # Màu nền trong suốt
        stroke_width=2,
        stroke_color="green",
        background_image=resized_image,  # Dùng ảnh đã resize
        update_streamlit=True,
        height=resized_image.height,
        width=resized_image.width,
        drawing_mode="rect",
        key="canvas",
    )

    # Nếu người dùng vẽ vùng crop
    cropped_image = resized_image  # Mặc định là ảnh gốc
    if canvas_result.json_data is not None:
        # Lấy tọa độ của vùng được vẽ
        points = canvas_result.json_data["objects"]
        if points:
            x0, y0 = points[0]["left"], points[0]["top"]
            x1, y1 = x0 + points[0]["width"], y0 + points[0]["height"]

            # Crop ảnh theo vùng đã chọn
            cropped_image = resized_image.crop((x0, y0, x1, y1))




# Thêm nút nhận dạng dưới ảnh
if uploaded_file and st.button("Nhận dạng"):
    # Thực hiện nhận dạng với ảnh đã crop hoặc ảnh gốc
    st.write("Đang nhận dạng...")
    temp_dir = "/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_image_path = os.path.join(temp_dir, uploaded_file.name)
    
    # Lưu ảnh đã crop hoặc ảnh gốc
    cropped_image.save(temp_image_path)

    # st.write(f"Ảnh đã được lưu tại: {temp_image_path}")
    # Tại đây bạn có thể gọi phương pháp nhận dạng của bạn, ví dụ:
    # recognition_result = some_recognition_function(cropped_image)
    
    # Ví dụ kết quả giả định:
    # recognition_result = "Kết quả nhận dạng: Chữ 'Thư pháp'"
    
    # Hiển thị kết quả nhận dạng
    # st.write(recognition_result)


    def run_method(method_name, image_path, gpu):
        result = ""
        command = None
        if task_type == "STR APPROACH":
            if method_name == "VietOCR":
                command = (
                    f"source /mlcv2/WorkingSpace/Personal/hamh/miniconda3/bin/activate VietOCR && "
                    f"cd /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/vietocr && "
                    f"CUDA_VISIBLE_DEVICES={gpu} python infer.py --image_path {image_path}"
                )
            elif method_name == "CCD":
                command = (
                    f"source /mlcv2/WorkingSpace/Personal/hamh/miniconda3/bin/activate CCD && "
                    f"cd /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/CCD/CCD_Ha && "
                    f"CUDA_VISIBLE_DEVICES={gpu} python infer.py -c Dino/configs/CCD_vision_model_ARD_base_ViCalligraphy.yaml "
                    f"--checkpoint saved_models/CCD_finetune_100epochs_ViCalligraphy_base_case_sensitive/best_accuracy.pth "
                    f"--image_path {image_path}"
                )
            elif method_name == "SMTR":
                command = (
                    f"source /mlcv2/WorkingSpace/Personal/hamh/miniconda3/bin/activate openocr && "
                    f"cd /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/OpenOCR && "
                    f"CUDA_VISIBLE_DEVICES={gpu} python tools/infer_my_rec.py --config /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/OpenOCR/configs/smtr/config.yml --image_path {image_path}"
                )
            elif method_name == "ABINet":
                command = (
                    f"source /mlcv2/WorkingSpace/Personal/hamh/miniconda3/bin/activate paddleocr-2.6.1 && "
                    f"cd /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/PaddleOCR && "
                    f"python3 tools/infer/predict_rec_demo.py --image_dir={image_path} "
                    f"--rec_model_dir='./inference/rec_r45_abinet/' --rec_algorithm='ABINet' "
                    f"--rec_image_shape='3,32,128' --rec_char_dict_path='./ppocr/utils/dict/vi_vietnam.txt'"
                )
            elif method_name == "PARSeq":
                command = (
                    f"source /mlcv2/WorkingSpace/Personal/hamh/miniconda3/bin/activate paddleocr-2.6.1 && "
                    f"cd /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/PaddleOCR && "
                    f"python3 tools/infer/predict_rec_demo.py --image_dir={image_path} "
                    f"--rec_model_dir='./inference/rec_parseq/' --rec_image_shape='3,32,128' "
                    f"--rec_algorithm='ParseQ' --rec_char_dict_path='./ppocr/utils/dict/vi_vietnam.txt' "
                    f"--max_text_length=25 --use_space_char=False"
                )
            elif method_name == "SRN":
                command = (
                    f"source /mlcv2/WorkingSpace/Personal/hamh/miniconda3/bin/activate paddleocr-2.6.1 && "
                    f"cd /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/PaddleOCR && "
                    f"python3 tools/infer/predict_rec_demo.py --image_dir={image_path} "
                    f"--rec_model_dir='./inference/rec_srn/' --rec_image_shape='1,64,256' "
                    f"--rec_algorithm='SRN' --rec_char_dict_path='ppocr/utils/dict/vi_vietnam.txt' "
                    f"--use_space_char=False"
                )
            elif method_name == "SVTR":
                command = (
                    f"source /mlcv2/WorkingSpace/Personal/hamh/miniconda3/bin/activate paddleocr-2.6.1 && "
                    f"cd /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/PaddleOCR && "
                    f"python3 tools/infer/predict_rec_demo.py --image_dir={image_path} "
                    f"--rec_model_dir='./inference/rec_svtr_tiny_stn_en/' --rec_algorithm='SVTR' --rec_image_shape='3,64,256' "
                    f"--rec_char_dict_path='ppocr/utils/dict/vi_vietnam.txt'"
                )
            elif method_name == "ViTSTR":
                command = (
                    f"source /mlcv2/WorkingSpace/Personal/hamh/miniconda3/bin/activate paddleocr-2.6.1 && "
                    f"cd /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/PaddleOCR && "
                    f"python3 tools/infer/predict_rec_demo.py --image_dir={image_path} "
                    f"--rec_model_dir='./inference/rec_vitstr/' --rec_algorithm='ViTSTR' --rec_image_shape='1,224,224' "
                    f"--rec_char_dict_path='./ppocr/utils/dict/vi_vietnam.txt'"
                )
        elif task_type == "CCD APPROACH":
            if method_name == "CCD":
                if training_data == "ViCalligraphy":
                    command = (
                        f"source /mlcv2/WorkingSpace/Personal/hamh/miniconda3/bin/activate CCD && "
                        f"cd /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/CCD/CCD_Ha && "
                        f"CUDA_VISIBLE_DEVICES={gpu} python infer.py -c Dino/configs/CCD_vision_model_ARD_base_ViCalligraphy.yaml "
                        f"--checkpoint saved_models/CCD_finetune_100epochs_ViCalligraphy_base_case_sensitive/best_accuracy.pth "
                        f"--image_path {image_path}"
                    )
                elif training_data == "ViCalligraphy_10000":
                    command = (
                        f"source /mlcv2/WorkingSpace/Personal/hamh/miniconda3/bin/activate CCD && "
                        f"cd /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/CCD/CCD_Ha && "
                        f"CUDA_VISIBLE_DEVICES={gpu} python infer.py -c Dino/configs/CCD_vision_model_ARD_base_ViCalligraphy.yaml "
                        f"--checkpoint saved_models/CCD_finetune_100epochs_ViCalligraphy_3000-VNI_7000-Unicode_base_case_sensitive/best_accuracy.pth "
                        f"--image_path {image_path}"
                    )
                elif training_data == "1m_ViCalligraphy":
                    command = (
                        f"source /mlcv2/WorkingSpace/Personal/hamh/miniconda3/bin/activate CCD && "
                        f"cd /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/CCD/CCD_Ha && "
                        f"CUDA_VISIBLE_DEVICES={gpu} python infer.py -c Dino/configs/CCD_vision_model_ARD_base_ViCalligraphy.yaml "
                        f"--checkpoint saved_models/CCD_finetune_1m_finetune_ViCalligraphy/best_accuracy.pth "
                        f"--image_path {image_path}"
                    )
                elif training_data == "1m_ViCalligraphy_10000":
                    command = (
                        f"source /mlcv2/WorkingSpace/Personal/hamh/miniconda3/bin/activate CCD && "
                        f"cd /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/CCD/CCD_Ha && "
                        f"CUDA_VISIBLE_DEVICES={gpu} python infer.py -c Dino/configs/CCD_vision_model_ARD_base_ViCalligraphy.yaml "
                        f"--checkpoint saved_models/CCD_finetune_1m_finetune_ViCalligraphy_3000-VNI_7000-Unicode/best_accuracy.pth "
                        f"--image_path {image_path}"
                    )
            elif method_name == "CCD-SLD":
                if training_data == "ViCalligraphy":
                    command = (
                        f"source /mlcv2/WorkingSpace/Personal/hamh/miniconda3/bin/activate CCD && "
                        f"cd /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/CCD/CCD_stroke && "
                        f"CUDA_VISIBLE_DEVICES={gpu} python infer.py -c /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/CCD/CCD_Ha/Dino/configs/CCD_vision_model_ARD_base_ViCalligraphy_stroke.yaml "
                        f"--checkpoint /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/CCD/CCD_Ha/saved_models/CCD_finetune_vicalligraphy_stroke_len_40_v1/best_accuracy.pth "
                        f"--image_path {image_path} --alg_name CCD-SLD"
                    )
                elif training_data == "ViCalligraphy_10000":
                    command = (
                        f"source /mlcv2/WorkingSpace/Personal/hamh/miniconda3/bin/activate CCD && "
                        f"cd /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/CCD/CCD_stroke && "
                        f"CUDA_VISIBLE_DEVICES={gpu} python infer.py -c /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/CCD/CCD_Ha/Dino/configs/CCD_vision_model_ARD_base_ViCalligraphy_stroke.yaml "
                        f"--checkpoint /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/CCD/CCD_Ha/saved_models/CCD_finetune_vicalligraphy_10000_stroke_len_40/best_accuracy.pth  "
                        f"--image_path {image_path} --alg_name CCD-SLD"
                    )
                elif training_data == "1m_ViCalligraphy":
                    command = (
                        f"source /mlcv2/WorkingSpace/Personal/hamh/miniconda3/bin/activate CCD && "
                        f"cd /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/CCD/CCD_stroke && "
                        f"CUDA_VISIBLE_DEVICES={gpu} python infer.py -c /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/CCD/CCD_Ha/Dino/configs/CCD_vision_model_ARD_base_ViCalligraphy_stroke.yaml "
                        f"--checkpoint /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/CCD/CCD_Ha/saved_models/CCD_finetune_1m_stroke_vicalligraphy_stroke_len_40_v2/best_accuracy.pth  "
                        f"--image_path {image_path} --alg_name CCD-SLD"
                    )
                elif training_data == "1m_ViCalligraphy_10000":
                    command = (
                        f"source /mlcv2/WorkingSpace/Personal/hamh/miniconda3/bin/activate CCD && "
                        f"cd /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/CCD/CCD_stroke && "
                        f"CUDA_VISIBLE_DEVICES={gpu} python infer.py -c /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/CCD/CCD_Ha/Dino/configs/CCD_vision_model_ARD_base_ViCalligraphy_stroke.yaml "
                        f"--checkpoint  /mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/CCD/CCD_Ha/saved_models/CCD_finetune_1m_stroke_vicalligraphy_10000_stroke_len_40/best_accuracy.pth "
                        f"--image_path {image_path} --alg_name CCD-SLD"
                    )
        try:
            # Chạy lệnh trong bash và thu thập kết quả
            subprocess.check_output(f"bash -c \"{command}\"", shell=True, text=True, stderr=subprocess.STDOUT)

            # Đọc kết quả từ file method.txt
            result_file = f"/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/one_image/results/{method_name}.txt"
            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    result = f.read().strip().split()[1]
            else:
                result = "Không tìm thấy kết quả!"
            return result

        except subprocess.CalledProcessError as e:
            return f"Lỗi khi chạy lệnh: {e.output}"
        except Exception as e:
            return f"Lỗi không xác định: {str(e)}"

    # Nếu người dùng chọn "Tất cả", chạy tất cả phương pháp và lưu kết quả
    if method == "Tất cả":
        results = {}
        for m in methods_list:
            result = run_method(m, temp_image_path, gpu)
            results[m] = result
        
        # Chuyển kết quả thành bảng ngang
        results_df = pd.DataFrame([results], columns=results.keys())
        st.subheader("Bảng so sánh kết quả")
        st.dataframe(results_df)

    else:
        # Chạy phương pháp đã chọn
        result = run_method(method, temp_image_path, gpu)
        st.subheader(f"Kết quả {method}: {result}")
        # st.image(uploaded_file, caption="Ảnh gốc", use_column_width=True)

    # Xóa ảnh tạm khi kết thúc
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

    clear_results_folder("/mlcv2/WorkingSpace/Personal/hamh/Ha/Methods/DemoSTR/one_image/results")
else:
    st.warning("Vui lòng tải lên một file ảnh!")
