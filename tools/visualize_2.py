from mmseg.apis.inference import init_model, inference_model, show_result_pyplot
import os
import cv2

# === Config and model checkpoint paths ===
config_file = 'work_dirs/bisenetv2_fcn_4xb4-40k_outback-375x600/bisenetv2_fcn_4xb4-40k_outback-375x600.py'
checkpoint_file = 'work_dirs/bisenetv2_fcn_4xb4-40k_outback-375x600/iter_10000.pth'

# === Path settings ===
#text_file = '../GANav-offroad/data/rellis/rellis_test_set_3.txt'  # lines like: subfolder/image_name (no extension)
#text_file = '../GANav-offroad/data/outback_DICTA_NW/test_171.txt'  # lines like: subfolder/image_name (no extension)
#base_folder = '../GANav-offroad/data/outback_DICTA_NW/test_data_3/rgb_final'       # folder containing RGB images
#base_folder = '../GANav-offroad/data/outback_DICTA_NW/test_data_3/depth_outback_4_env'
#save_dir = 'work_dirs/iter_50K_rgb_outback_ori_on_rellis_2000'                              # output folder for visual results

text_file = '../GANav-offroad/data/rellis/rellis_test_set_3.txt'  # lines like: subfolder/image_name (no extension)
#base_folder = '../GANav-offroad/data/rellis/RELLIS_depth_set_3_new'       # folder containing RGB images
base_folder = '../GANav-offroad/data/rellis/00003_test_rgb'       # folder containing RGB images

save_dir = 'work_dirs/iter_10K_rgb_complete_thesis'  

# === Initialize the model ===
model = init_model(config_file, checkpoint_file, device='cuda:0')

# === Read filenames from txt file ===
with open(text_file, 'r') as f:
    lines = [line.strip() for line in f if line.strip()]

# === Inference loop ===
for line in lines:
    rel_path = line + '.png'  # assumes .png image format
    full_path = os.path.join(base_folder, rel_path)

    if not os.path.exists(full_path):
        print(f"[WARN] File not found: {full_path}")
        continue

    # Load RGB image
    img = cv2.imread(full_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for inference

    # Inference
    result = inference_model(model, img)

    # Prepare output path
    output_path = os.path.join(save_dir, line + '.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save visualization
    show_result_pyplot(
        model,
        img,
        result,
        opacity=1,
        show=False,
        out_file=output_path
    )

print("\n✅ Inference complete. Results saved to:", save_dir)
