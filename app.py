import streamlit as st
import os
import numpy as np
import torch
import cv2
from PIL import Image
import kornia as K
import kornia.feature as KF
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torchvision
from torchvision.transforms import functional as F
import json
import tempfile
import pandas as pd
import gc

st.set_page_config(layout="wide", page_title="VO Benchmark App")

# --- Initialize Session State ---
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
    st.session_state.speeds = []
    st.session_state.distances = []
    st.session_state.total_frames = 0
    st.session_state.masked_video_path = None

# ==============================================================================
# STEP 1: AI MODEL LOADING (CACHED)
# ==============================================================================

# We now cache each model separately
@st.cache_resource
def load_maskrcnn():
    st.toast("Loading Mask R-CNN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT').to(device).eval()
    print("✅ Mask R-CNN loaded.")
    return model, device

@st.cache_resource
def load_vo_models():
    st.toast("Loading ZoeDepth and LoFTR models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zoe_processor = AutoImageProcessor.from_pretrained("Intel/zoedepth-nyu")
    zoe_model = AutoModelForDepthEstimation.from_pretrained("Intel/zoedepth-nyu").to(device).eval()
    loftr_model = KF.LoFTR(pretrained='outdoor').to(device).eval()
    print("✅ ZoeDepth and LoFTR loaded.")
    return zoe_processor, zoe_model, loftr_model, device

# ==============================================================================
# STEP 2: HELPER FUNCTIONS (THE BACKEND)
# ==============================================================================

def get_camera_intrinsics(video_path, custom_intrinsics_file):
    """
    Loads camera intrinsics. If a custom file is provided, it's used.
    Otherwise, it estimates them from the video's width.
    """
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if custom_intrinsics_file is not None:
        st.toast("Loading custom camera intrinsics...")
        try:
            intrinsics_data = json.load(custom_intrinsics_file)
            K_CAM = np.array(intrinsics_data['K'])
        except Exception as e:
            st.error(f"Error reading intrinsics file: {e}. Using default.")
            FOCAL_LENGTH = W
            K_CAM = np.array([[FOCAL_LENGTH, 0, W / 2], [0, FOCAL_LENGTH, H / 2], [0, 0, 1]], dtype=np.float32)
    else:
        st.toast("Using default estimated camera intrinsics.")
        FOCAL_LENGTH = W
        K_CAM = np.array([[FOCAL_LENGTH, 0, W / 2], [0, FOCAL_LENGTH, H / 2], [0, 0, 1]], dtype=np.float32)

    return K_CAM, W, H, FPS, total_frames

def zoedepth_predict(processor, model, img_bgr, device):
    pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    h, w = img_bgr.shape[:2]
    predicted_depth = torch.nn.functional.interpolate(
        out.predicted_depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False
    ).squeeze()
    return predicted_depth.cpu().numpy()

# --- STAGE 1: MASKING PIPELINE ---
def run_masking_stage(video_path, W, H, FPS, total_frames, mask_model, device):
    """
    Runs ONLY the masking part of the pipeline.
    """
    st.info("Starting Stage 1: Masking video...")
    
    # We will write the masked video to a temporary file
    temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_video_path = temp_video_file.name
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, FPS, (W, H))
    cap = cv2.VideoCapture(video_path)
    
    progress_bar = st.progress(0, text="Stage 1: Masking video...")

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = F.to_tensor(rgb_frame).to(device)
        with torch.no_grad():
            prediction = mask_model([img_tensor])
        
        combined_mask = np.zeros((H, W), dtype="uint8")
        for i in range(len(prediction[0]['labels'])):
            if prediction[0]['labels'][i] == 1 and prediction[0]['scores'][i] > 0.5: # 1 = person
                person_mask = (prediction[0]['masks'][i, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
                combined_mask = cv2.bitwise_or(combined_mask, person_mask)
        
        frame[combined_mask == 255] = 0
        out_writer.write(frame)
        
        progress_bar.progress((frame_idx + 1) / total_frames, text=f"Stage 1: Masking frame {frame_idx+1}/{total_frames}")

    cap.release()
    out_writer.release()
    progress_bar.empty()
    st.success("✅ Stage 1: Masking complete.")
    return output_video_path

# --- STAGE 2: ODOMETRY PIPELINE ---
def run_odometry_stage(video_path, K_CAM, W, H, FPS, total_frames, zoe_processor, zoe_model, loftr_model, device):
    """
    Runs ONLY the odometry part on the pre-masked video.
    """
    st.info("Starting Stage 2: Calculating odometry...")
    
    R_total, T_total = np.eye(3), np.zeros((3, 1))
    trajectory = [T_total.flatten()]
    speeds = []
    dt = 1.0 / FPS if FPS > 0 else 0.033
    
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame_bgr = cap.read()
    if not ret:
        raise Exception("Could not read the first frame.")
        
    progress_bar = st.progress(0, text="Stage 2: Calculating odometry...")

    for frame_idx in range(total_frames - 1):
        ret, curr_frame_bgr = cap.read()
        if not ret:
            break
        
        try:
            depth_map = zoedepth_predict(zoe_processor, zoe_model, prev_frame_bgr, device)
            img0_tensor = K.image_to_tensor(cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY), False).float() / 255.
            img1_tensor = K.image_to_tensor(cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2GRAY), False).float() / 255.
            
            with torch.no_grad():
                correspondences = loftr_model({"image0": img0_tensor.to(device), "image1": img1_tensor.to(device)})

            mask = correspondences['confidence'].cpu().numpy() > 0.9
            if mask.sum() < 10:
                prev_frame_bgr = curr_frame_bgr.copy()
                speeds.append(0)
                continue

            mkpts0 = correspondences['keypoints0'].cpu().numpy()[mask]
            mkpts1 = correspondences['keypoints1'].cpu().numpy()[mask]

            pts_3d, pts_2d = [], []
            for pt0, pt1 in zip(mkpts0, mkpts1):
                x, y = int(pt0[0]), int(pt0[1])
                d = depth_map[y, x]
                if d > 0.1:
                    X = (x - K_CAM[0, 2]) * d / K_CAM[0, 0]
                    Y = (y - K_CAM[1, 2]) * d / K_CAM[1, 1]
                    pts_3d.append([X, Y, d])
                    pts_2d.append(pt1)

            if len(pts_3d) < 10:
                prev_frame_bgr = curr_frame_bgr.copy()
                speeds.append(0)
                continue

            success, rvec, tvec, _ = cv2.solvePnPRansac(np.array(pts_3d), np.array(pts_2d), K_CAM, None)
            if not success:
                prev_frame_bgr = curr_frame_bgr.copy()
                speeds.append(0)
                continue

            R, _ = cv2.Rodrigues(rvec)
            R_inv, t_inv = R.T, -R.T @ tvec
            
            distance = np.linalg.norm(t_inv)
            speeds.append(distance / dt)
            
            T_total = T_total + R_total @ t_inv
            R_total = R_inv @ R_total
            trajectory.append(T_total.flatten())
            
        except Exception as e:
            print(f"Error on frame {frame_idx}: {e}")
            speeds.append(0) # Append 0 speed if an error occurs

        prev_frame_bgr = curr_frame_bgr.copy()
        
        progress_bar.progress((frame_idx + 1) / (total_frames - 1), text=f"Stage 2: Processing frame {frame_idx+1}/{total_frames}")

    cap.release()
    progress_bar.empty()
    st.success("✅ Stage 2: Odometry complete.")
    
    # Calculate cumulative distance
    distances = [np.linalg.norm(trajectory[i] - trajectory[i-1]) for i in range(1, len(trajectory))]
    cumulative_distance = np.cumsum(distances)
    
    # Save results to session state
    st.session_state.speeds = [0.0] + list(speeds)
    st.session_state.distances = [0.0, 0.0] + list(cumulative_distance) # Pad for 2 frames
    st.session_state.processing_complete = True


# ==============================================================================
# STEP 3: STREAMLIT UI LAYOUT
# ==============================================================================

st.title("📹 Visual Odometry with Dynamic Masking")
st.write("This app runs a visual odometry pipeline that first masks people (dynamic objects) "
         "before estimating camera speed and trajectory.")

# --- Sidebar for Uploads and Control ---
st.sidebar.header("Controls")
video_file = st.sidebar.file_uploader("1. Upload Your Video", type=["mp4", "mov", "avi"])

with st.sidebar.expander("2. Upload Custom Intrinsics (Optional)"):
    st.info('Upload a `.json` file with your camera matrix. If omitted, '
            'an estimation will be used.')
    st.code("""
    {
        "K": [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]
    }
    """)
    intrinsics_file = st.file_uploader("Upload Intrinsics (.json)", type=["json"])

process_button = st.sidebar.button("Start Processing", disabled=(video_file is None))

# --- Main App Display Area ---
if process_button:
    # Reset state if we're processing a new video
    st.session_state.processing_complete = False
    
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    video_path = tfile.name
    
    # Get camera properties (handling custom intrinsics)
    K_CAM, W, H, FPS, total_frames = get_camera_intrinsics(video_path, intrinsics_file)
    
    # --- RUN STAGE 1: MASKING ---
    mask_model, device = load_maskrcnn()
    masked_video_path = run_masking_stage(video_path, W, H, FPS, total_frames, mask_model, device)
    
    # --- Manually clear Mask R-CNN from memory ---
    del mask_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # --- RUN STAGE 2: ODOMETRY ---
    zoe_processor, zoe_model, loftr_model, device = load_vo_models()
    run_odometry_stage(masked_video_path, K_CAM, W, H, FPS, total_frames,
                       zoe_processor, zoe_model, loftr_model, device)
    
    # --- Manually clear VO models from memory ---
    del zoe_processor, zoe_model, loftr_model
    gc.collect()
    torch.cuda.empty_cache()

    # Save the path to the masked video for the dashboard
    st.session_state.masked_video_path = masked_video_path
    
    # Clean up the original uploaded video temp file
    os.remove(video_path)
    
    # Trigger a re-run of the script to show the results
    st.rerun()

# --- Display Results Dashboard (if processing is complete) ---
if st.session_state.processing_complete:
    st.header("Metrics Dashboard")
    
    total_frames = st.session_state.total_frames
    speeds = st.session_state.speeds
    distances = st.session_state.distances
    
    # Fix for length mismatch
    max_len = st.session_state.total_frames
    if len(speeds) < max_len:
        speeds.extend([0.0] * (max_len - len(speeds)))
    if len(distances) < max_len:
        distances.extend([distances[-1]] * (max_len - len(distances)))
    
    speeds = speeds[:max_len]
    distances = distances[:max_len]

    # Main slider
    frame_idx = st.slider("Scrub Through Video Frames", 0, total_frames - 1, 0)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("SPEEDOMETER (m/s)", f"{speeds[frame_idx]:.2f}")
    with col2:
        st.metric("ODOMETER (meters traveled)", f"{distances[frame_idx]:.2f}")
    
    # Display the processed video
    video_bytes = open(st.session_state.masked_video_path, 'rb').read()
    st.video(video_bytes)
    
    with st.expander("Show Raw Data"):
        df = pd.DataFrame({
            'Frame': list(range(total_frames)),
            'Speed (m/s)': speeds,
            'Distance (m)': distances
        })
        st.dataframe(df)

else:
    st.info("Please upload a video and click 'Start Processing' in the sidebar.")