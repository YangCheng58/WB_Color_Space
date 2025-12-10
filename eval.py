import logging
import os
import torch
import numpy as np
import cv2
from PIL import Image
from scipy.io import loadmat
from os.path import join, basename, exists, splitext
from glob import glob  # <--- Key: Added glob module
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from skimage import color
from sklearn.linear_model import LinearRegression
from torchvision import transforms
from arch.DCLAN import DCLAN
import argparse

def outOfGamutClipping(I):
    I[I > 1] = 1
    I[I < 0] = 0
    return I

def kernelP(I):
    """ Polynomial feature expansion """
    return (np.transpose((I[:, 0], I[:, 1], I[:, 2], I[:, 0] * I[:, 1], I[:, 0] * I[:, 2],
                          I[:, 1] * I[:, 2], I[:, 0] * I[:, 0], I[:, 1] * I[:, 1],
                          I[:, 2] * I[:, 2], I[:, 0] * I[:, 1] * I[:, 2],
                          np.repeat(1, np.shape(I)[0]))))

def get_mapping_func(image1, image2):
    """ Compute mapping matrix """
    image1 = np.reshape(image1, [-1, 3])
    image2 = np.reshape(image2, [-1, 3])
    m = LinearRegression().fit(kernelP(image1), image2)
    return m

def apply_mapping_func(image, m):
    """ Apply mapping matrix """
    sz = image.shape
    image = np.reshape(image, [-1, 3])
    result = m.predict(kernelP(image))
    result = np.reshape(result, [sz[0], sz[1], sz[2]])
    return result

def deep_wb_inference(image, net_awb, device='cpu', s=640):
    """
    Main Inference Pipeline: Resize -> Model -> Mapping -> Original Size
    """
    # 1. Preprocessing
    image_np = np.array(image).astype(np.float32) / 255.0
    
    ratio = s / max(image.size)
    new_w = round(image.width * ratio)
    new_h = round(image.height * ratio)
    if new_w % 128 != 0: new_w += 128 - (new_w % 128)
    if new_h % 128 != 0: new_h += 128 - (new_h % 128)
    
    image_resized = image.resize((new_w, new_h), Image.BILINEAR)
    image_resized_np = np.array(image_resized).astype(np.float32) / 255.0
    
    img_tensor = torch.from_numpy(np.transpose(image_resized_np, (2, 0, 1))).unsqueeze(0).to(device, dtype=torch.float32)

    # 2. Model Inference
    net_awb.eval()
    with torch.no_grad():
        output = net_awb(img_tensor)
        if isinstance(output, (tuple, list)):
            output_awb = output[0]
        else:
            output_awb = output
    
    output_awb = output_awb.clamp(0, 1)
    output_awb_np = output_awb.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    # 3. Compute and Apply Mapping
    m_awb = get_mapping_func(image_resized_np, output_awb_np)
    result_high_res = apply_mapping_func(image_np, m_awb)
    result_high_res = outOfGamutClipping(result_high_res)

    return result_high_res


def calc_metrics(pred, gt):
    # MSE
    mse = np.mean(((pred - gt) * 255.0) ** 2)
    
    # MAE
    pred_vec = pred.reshape(-1, 3)
    gt_vec = gt.reshape(-1, 3)
    source_norm = np.sqrt(np.sum(np.power(pred_vec, 2), 1))
    target_norm = np.sqrt(np.sum(np.power(gt_vec, 2), 1))
    norm = source_norm * target_norm
    inds = norm != 0
    angles = np.sum(pred_vec[inds] * gt_vec[inds], 1) / norm[inds]
    angles = np.clip(angles, -1, 1)
    mae = np.mean(np.arccos(angles) * 180 / np.pi)
    
    # Delta E
    pred_lab = color.rgb2lab(pred)
    gt_lab = color.rgb2lab(gt)
    delta_e = np.mean(color.deltaE_ciede2000(pred_lab, gt_lab))
    
    return mse, mae, delta_e


def get_image_pairs(config):
    pairs = []
    dataset_type = config['type']
    if dataset_type == 'mat_file':
        mat_path = config['mat_file']
        img_root = config['img_root']
        # Set1 GTs are usually in the same directory, or specified via gt_root
        gt_root = config.get('gt_root', img_root)
        
        if not exists(mat_path):
            logging.error(f"Mat file not found: {mat_path}")
            return []
            
        logging.info(f"Loading Mat: {basename(mat_path)}")
        try:
            # 1. Load Mat
            mat_data = loadmat(mat_path)
            patterns = mat_data['validation'] 
            
            # 2. Collect all images (Handle glob wildcards)
            all_input_files = []
            for i in range(len(patterns)):
                rel_pattern = patterns[i][0][0] # e.g., "folder/*.jpg"
                full_pattern = join(img_root, rel_pattern)
                found = glob(full_pattern)
                all_input_files.extend(found)
            
            logging.info(f"Mat file matched {len(all_input_files)} images, starting to match GT...")

            # 3. Iterate images and match GT (Core Logic)
            for img_file in all_input_files:
                f_name = basename(img_file)
                
                # --- [Start] Set1 filename matching logic ---
                gt_ext = 'G_AS.png'
                parts = f_name.split('_')
                # e.g.: '8D5U5526_001_input.jpg' -> ['8D5U5526', '001', 'input.jpg']
                
                base_name = ''
                # Keep first n-2 parts
                if len(parts) >= 2:
                    for k in range(len(parts) - 2):
                        base_name = base_name + parts[k] + '_'
                    # Result: '8D5U5526_001_'
                    
                    gt_name = base_name + gt_ext
                    # Result: '8D5U5526_001_G_AS.png'
                else:
                    # If filename is short, doesn't fit rule, use original name or skip
                    gt_name = f_name
                # --- [End] Set1 filename matching logic ---
                
                gt_path = join(gt_root, gt_name)
                
                if exists(gt_path):
                    pairs.append((img_file, gt_path))
                # else: logging.warning(f"Set1 GT not found: {gt_name}")
                    
        except Exception as e:
            logging.error(f"Set1 (Mat) parsing failed: {e}")

    elif dataset_type == 'folder':
        input_dir = config['input_dir']
        gt_dir = config['gt_dir']
        style = config.get('match_style', 'direct')
        
        if not exists(input_dir):
            logging.error(f"Input directory not found: {input_dir}")
            return []

        valid_exts = {'.jpg', '.png', '.jpeg', '.bmp'}
        input_files = sorted([f for f in os.listdir(input_dir) if splitext(f)[1].lower() in valid_exts])
        
        for f_name in input_files:
            inp_path = join(input_dir, f_name)
            
            if style == 'cube':
                # Cube rule: "001_input.png" -> "001.JPG"
                gt_name = f_name.split("_")[0] + ".JPG"
            else:
                # Set2 rule: Same name
                gt_name = f_name
                
            gt_path = join(gt_dir, gt_name)
            
            if exists(gt_path):
                pairs.append((inp_path, gt_path))
            
    return pairs

def worker(args):
    inp_path, gt_path, net, device = args
    try:
        # Read image
        img_pil = Image.open(inp_path).convert('RGB')
        gt_pil = Image.open(gt_path).convert('RGB')
        
        # Inference
        pred_np = deep_wb_inference(img_pil, net, device)
        
        # GT Processing
        gt_np = np.array(gt_pil).astype(np.float32) / 255.0
        
        # Force size alignment
        if pred_np.shape != gt_np.shape:
            gt_np = cv2.resize(gt_np, (pred_np.shape[1], pred_np.shape[0]))
            
        # Calculate Metrics
        mse, mae, de = calc_metrics(pred_np, gt_np)
        return {'file': basename(inp_path), 'mse': mse, 'mae': mae, 'de': de}
    except Exception as e:
        return {'error': str(e)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    parser = argparse.ArgumentParser(description="Evaluation Script for DCLAN")
    # Core arguments
    parser.add_argument('--dataset', type=str, required=True, choices=['Set1', 'Set2', 'Cube'], help='Dataset to evaluate')
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth model weights')
    parser.add_argument('--device', type=str, default='cuda:1', help='Device (e.g., cuda:0 or cpu)')
    
    # Set1 specific arguments
    parser.add_argument('--data_root', type=str, default='./dataset/Set1_all', help='Root for Set1 images')
    parser.add_argument('--split_file', type=str, default='./folds/fold3_.mat', help='Path to fold mat file')
    
    # Set2/Cube specific arguments
    parser.add_argument('--input_dir', type=str, help='Input directory for Set2/Cube')
    parser.add_argument('--gt_dir', type=str, help='GT directory for Set2/Cube')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # -------------------------------------------------------------
    # 🎛️ Dataset Configuration
    # -------------------------------------------------------------
    config = {}
    if args.dataset == 'Set1':
        config = {
            'type': 'mat_file',
            'img_root': args.data_root,
            'gt_root': args.data_root,
            'mat_file': args.split_file
        }
    elif args.dataset == 'Set2':
        config = {
            'type': 'folder',
            'input_dir': args.input_dir,
            'gt_dir': args.gt_dir,
            'match_style': 'direct'
        }
    elif args.dataset == 'Cube':
        config = {
            'type': 'folder',
            'input_dir': args.input_dir,
            'gt_dir': args.gt_dir,
            'match_style': 'cube'
        }

    # -------------------------------------------------------------
    # 🚀 Model Loading & Execution
    # -------------------------------------------------------------
    logging.info(f"=== Task Started: {args.dataset} | Model: {basename(args.model_path)} ===")

    net = DCLAN()
    
    if exists(args.model_path):
        state = torch.load(args.model_path, map_location=device)
        # Handle potential 'module.' prefix
        if 'state_dict' in state: state = state['state_dict']
        if list(state.keys())[0].startswith('module.'):
            state = {k.replace('module.', ''): v for k, v in state.items()}
        net.load_state_dict(state)
        net.to(device)
        net.eval()
        logging.info("Model weights loaded successfully")
    else:
        logging.error(f"Weight file not found: {args.model_path}")
        exit()

    # 2. Match Data
    pairs = get_image_pairs(config)
    
    if not pairs:
        logging.error("No matching image pairs found. Please check the path arguments!")
        exit()
        
    logging.info(f"Successfully matched image pairs: {len(pairs)}")

    # 3. Execute Evaluation
    results = []
    task_args = [(p[0], p[1], net, device) for p in pairs]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        for res in tqdm(executor.map(worker, task_args), total=len(pairs), desc="Inferencing"):
            if 'error' not in res:
                results.append(res)
            else:
                logging.error(f"Skipped: {res['error']}")

    # 4. Statistical Results
    if results:
        mses = [r['mse'] for r in results]
        maes = [r['mae'] for r in results]
        des  = [r['de']  for r in results]

        # calculate Q1, Median, Q3
        mse_q = np.percentile(mses, [25, 50, 75])
        mae_q = np.percentile(maes, [25, 50, 75])
        de_q  = np.percentile(des,  [25, 50, 75])

        print("\n" + "="*65)
        print(f"📊 Dataset: {args.dataset} (Total: {len(results)})")
        print("-" * 65)
        print(f"{'Metric':<7} | {'Mean':<8} | {'Q1 (25%)':<8} | {'Med (50%)':<8} | {'Q3 (75%)':<8}")
        print("-" * 65)
        
        print(f" {'MSE':<6} | {np.mean(mses):<8.4f} | {mse_q[0]:<8.4f} | {mse_q[1]:<8.4f} | {mse_q[2]:<8.4f}")
        print(f" {'MAE':<6} | {np.mean(maes):<8.4f} | {mae_q[0]:<8.4f} | {mae_q[1]:<8.4f} | {mae_q[2]:<8.4f}")
        print(f" {'dE00':<6} | {np.mean(des):<8.4f} | {de_q[0]:<8.4f} | {de_q[1]:<8.4f} | {de_q[2]:<8.4f}")
        print("="*65 + "\n")

        # Record outliers
        outliers = [r for r in results if r['mse'] > 500]
        if outliers:
            outfile = f"outliers_{args.dataset}.txt"
            with open(outfile, "w") as f:
                for o in outliers:
                    f.write(f"{o['file']} | MSE:{o['mse']:.2f}\n")
            logging.info(f"Saved {len(outliers)} outliers to {outfile}")