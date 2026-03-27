from mmseg.apis.inference import init_model, inference_model, show_result_pyplot
import numpy as np

# === Config and model checkpoint paths ===
config_file = 'work_dirs/bisenetv2_fcn_4xb4-40k_outback-375x600/bisenetv2_fcn_4xb4-40k_outback-375x600_100K_16_batch.py'
checkpoint_file = 'work_dirs/bisenetv2_fcn_4xb4-40k_outback-375x600/iter_100000_100K_16_batch.pth'

# === Image path and output paths ===
#img_path = 'rgb_04268.png' #my comment
npy_path='rgb_1048.npy' #my code
out_file = 'work_dirs/vis_preds/result_image_NW_67.png'
save_dir = 'work_dirs/vis_preds'

def main():
    
    #my code->
        # Load 6-channel image from .npy
    img = np.load(npy_path)  # shape: (H, W, 6)
    print(f"Loaded .npy image shape: {img.shape}")

    # Wrap in results dict for transforms
    data = dict(
        img=img,
        img_shape=img.shape[:2],
        ori_shape=img.shape[:2],
        img_path=npy_path,
        inputs=img  # For compatibility with newer MMEngine
    )
    # my code^^^"""
    #'''
    # Initialize the model
    model = init_model(config_file, checkpoint_file, device='cuda:0')

    # Run inference
    #result = inference_model(model, img_path) # my comment
    result = inference_model(model, img) #my code
    # Visualize and save result
    show_result_pyplot(
        model, 
        #img_path, # my comment
        img, #my code
        result, 
        opacity=1, 
        show=False, 
        out_file=out_file,
        save_dir=save_dir
    )

if __name__ == '__main__':
    main()
