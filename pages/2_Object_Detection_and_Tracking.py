import streamlit as st
import os
import glob
from natsort import natsorted

from detection_logic import run_detection_and_tracking, sorted_by_frame_index
from visualization import create_3d_figure, generate_tracking_animation

st.set_page_config(layout="wide", page_title="Object Detection and Tracking")

st.title("📦 Object Detection and Tracking")

# Initialize session state
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None

# --- Input and Output Paths ---
st.subheader("📁 Input and Output")

filtered_pcd_dir = st.text_input(
    "Enter the path to the FILTERED PCD files (for detection):",
    value=r"outputs/background_filtering"
)

original_pcd_dir = st.text_input(
    "Enter the path to the ORIGINAL PCD files (for visualization):",
    value=r"data/point_clouds/cropped/cropped_pcd"
)

output_dir = "outputs/object_detection"

# --- Parameters ---
st.subheader("⚙️ Algorithm Parameters")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Clustering and Detection")
    dbscan_eps = st.slider("DBSCAN Epsilon (eps)", 0.1, 5.0, 2.0, 0.1, help="Controls cluster density.")
    min_cluster_pts = st.slider("Min Cluster Points", 1, 50, 1, 1, help="Minimum points to form a cluster.")
    min_hits = st.slider("Min Temporal Hits", 1, 10, 2, 1, help="Frames a candidate must exist to be confirmed.")

with col2:
    st.markdown("#### Tracking and Association")
    fps = st.slider("Frames Per Second (FPS)", 1.0, 30.0, 10.0, 0.5, help="Data frame rate for velocity calculation.")
    max_missed = st.slider("Max Missed Frames", 0, 20, 5, 1, help="Frames to keep a track alive without detection.")
    moving_speed_thresh = st.slider("Moving Speed Threshold (m/s)", 0.0, 10.0, 3.0, 0.1, help="Speed above which an object is 'moving'.")
    roi_abs_y = st.slider("ROI Absolute Y (m)", 5.0, 100.0, 40.0, 1.0, help="Y-coordinate processing range.")

st.markdown("#### Visualization")
col_v1, col_v2, col_v3 = st.columns(3)
with col_v1:
    eye_x = st.number_input("Camera Eye X", value=0.8, step=0.05)
with col_v2:
    eye_y = st.number_input("Camera Eye Y", value=0.8, step=0.05)
with col_v3:
    eye_z = st.number_input("Camera Eye Z", value=0.8, step=0.05)

max_frames_to_animate = st.number_input("Max Frames to Animate (0 for all)", min_value=0, max_value=2000, value=0, help="Limit the number of frames to process for the animation to save time.")

st.divider()

if st.button("🚀 Start Detection and Tracking", use_container_width=True):
    st.session_state.detection_results = None # Reset results

    # --- Validate Paths and Scan Files ---
    if not all(os.path.isdir(p) for p in [filtered_pcd_dir, original_pcd_dir]):
        st.error("One of the input directories is invalid. Please check the paths.")
    else:
        filtered_files = sorted_by_frame_index(glob.glob(os.path.join(filtered_pcd_dir, "*.pcd")))
        original_files = sorted_by_frame_index(glob.glob(os.path.join(original_pcd_dir, "*.pcd")))

        # --- File Count Validation ---
        if not filtered_files:
            st.error("No PCD files found in the filtered directory.")
        elif len(filtered_files) != len(original_files):
            st.error(f"PCD file count mismatch: {len(filtered_files)} filtered vs {len(original_files)} original files.")
        else:
            params = {
                'dbscan_eps': dbscan_eps, 'min_cluster_pts': min_cluster_pts, 'min_hits': min_hits,
                'roi_abs_y': roi_abs_y, 'yaw_bias_deg': -90.0,
                'fps': fps, 'max_missed': max_missed, 'moving_speed_thresh': moving_speed_thresh,
                'merge_dist': 2.5, 'yaw_merge_deg': 15.0, 'truck_len_thresh': 7.0, 'truck_merge_dist': 10.0,
            }
            st.info(f"Processing {len(filtered_files)} files from: {filtered_pcd_dir}...")
            progress_bar = st.progress(0, text="Starting...")
            def update_progress(current, total, message):
                progress_bar.progress(current / total, text=f"{message}: {current}/{total} frames")
            try:
                results, error_message = run_detection_and_tracking(filtered_pcd_dir, output_dir, params, update_progress)
                if error_message:
                    st.error(error_message)
                else:
                    # Add the sorted lists of files to the results for the UI
                    results['original_pcd_files'] = original_files
                    results['params'] = params # Pass params to visualization
                    st.session_state.detection_results = results
                    st.success(f"✅ Processing finished! Found {len(results['pcd_files'])} frames. Use the slider below to visualize.")
                progress_bar.empty()
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                progress_bar.empty()

# --- Visualization Section --- 
if st.session_state.detection_results:
    st.divider()
    st.subheader("🖼️ Interactive Visualization")
    results = st.session_state.detection_results

    # Update camera eye in results so visualization uses current UI settings
    results['camera_eye'] = {'x': eye_x, 'y': eye_y, 'z': eye_z}
    
    frame_idx = st.slider("Select Frame", 0, len(results['pcd_files']) - 1, 0)

    st.subheader("3D Point Cloud View")
    original_pcd_path = results['original_pcd_files'][frame_idx]
    if not os.path.exists(original_pcd_path):
        st.error(f"Original PCD file not found for this frame: {original_pcd_path}")
    else:
        fig = create_3d_figure(results, frame_idx, original_pcd_path)
        st.plotly_chart(fig, use_container_width=True, height=800)

    # --- Animation Generation Section ---
    st.divider()
    st.subheader("🎥 Tracking Animation")

    animation_path = os.path.join(output_dir, "tracking_animation.gif")

    # Always try to display the animation if it exists in the output directory
    if os.path.exists(animation_path):
        st.image(animation_path, caption="Tracking animation (last generated)")
    else:
        st.info("No animation generated yet. Click the button below to create one.")

    if st.button("🎬 Generate / Update Tracking Animation", use_container_width=True):
        st.info("Starting animation generation... This may take a few minutes.")
        progress_bar = st.progress(0, text="Initializing...")

        def animation_progress_callback(current, total):
            progress_bar.progress(current / total, text=f"Processing frame {current}/{total}")

        try:
            generate_tracking_animation(results, animation_path, animation_progress_callback, max_frames=max_frames_to_animate)
            st.success(f"✅ Animation successfully saved to: {animation_path}")
            st.balloons()
            # Force a rerun to display the new animation immediately
            st.rerun()
        except Exception as e:
            st.error(f"An error occurred during animation generation: {e}")
        finally:
            progress_bar.empty()
