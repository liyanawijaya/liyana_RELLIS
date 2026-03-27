import os
import numpy as np
from mmseg.apis import inference_model, init_model, show_result_pyplot

# === Paths ===
config_file = 'work_dirs//bisenetv2_fcn_4xb4-40k_outback-375x600//bisenetv2_fcn_4xb4-40k_outback-375x600.py'
checkpoint_file = 'work_dirs//bisenetv2_fcn_4xb4-40k_outback-375x600//iter_10000.pth'

text_file = '../GANav-offroad/data/outback_DICTA_NW/test_171.txt'  # lines like: subfolder/image_name (no extension)
base_folder = '../GANav-offroad/data/outback_DICTA_NW/test_data_3/rgbd_stacked_test_1'       # folder containing RGB images
#save_dir = 'work_dirs/iter_50K_rgb_stacked_outback_on_rellis_2000'     

#text_file = '../GANav-offroad/data/rellis/rellis_test_set_3.txt'  # lines like: subfolder/image_name (no extension)
#base_folder = '../GANav-offroad//data//rellis//00003_test_rgbd'       # folder containing RGB images
##save_dir = 'work_dirs/iter_50K_rgb_stacked_outback_on_rellis_2000'   

#text_file = '../GANav-offroad//data//rellis//rellis_test_set_3.txt'  # lines like: subfolder/image_name (no extension)
#base_folder = '../GANav-offroad/data/rellis/RELLIS_depth_set_3_new'       # folder containing RGB images
#base_folder = '../GANav-offroad/data/rellis/00003_test_rgb'       # folder containing RGB images

save_dir = 'work_dirs//iter_10K_model_3_complex_thesis_outback_test_4'

# === Setup model ===
model = init_model(config_file, checkpoint_file, device='cuda:0')

# === Load filenames ===
with open(text_file, 'r') as f:
    lines = [line.strip() for line in f.readlines() if line.strip()]

for line in lines:
    rel_path = line + '.npy'  # e.g., area1/rgb_00123.npy
    full_path = os.path.join(base_folder, rel_path)
    
    if not os.path.exists(full_path):
        print(f"[WARN] File not found: {full_path}")
        continue

    img = np.load(full_path)  # shape: (H, W, C)

    # Inference
    result = inference_model(model, img)

    # Save
    output_path = os.path.join(save_dir, line + '.png')  # result as PNG
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    show_result_pyplot(
        model,
        img,
        result,
        opacity=1,
        show=False,
        out_file=output_path,
        save_dir=None  # Not needed when out_file is given
    )

print("\n✅ Inference complete. Results saved.")
