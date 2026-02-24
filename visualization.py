import os
import numpy as np
import plotly.graph_objects as go
from detection_logic import load_points_from_pcd

def create_3d_figure(results, frame_index_to_render, original_pcd_path, camera_dict=None):
    """
    Creates an interactive 3D Plotly figure for a given frame.
    """
    fig = go.Figure()

    # 1. Add Original Point Cloud
    points = load_points_from_pcd(original_pcd_path)
    fig.add_trace(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers', name='Original Point Cloud',
        marker=dict(size=1, color='grey', opacity=0.5)
    ))

    # 2. Add Road Polygon
    road_poly = results['road_poly']
    if road_poly.geom_type == 'Polygon': polys = [road_poly]
    else: polys = road_poly.geoms
    for poly in polys:
        x, y = poly.exterior.xy
        x_np = np.array(x)
        y_np = np.array(y)
        fig.add_trace(go.Scatter3d(
            x=x_np, y=y_np, z=np.full_like(x_np, -7.5),
            mode='lines', name='Road',
            line=dict(color='green', width=4)
        ))

    # 3. Add Detections with ID and Speed Text
    det_frames = results['det_frames']
    current_dets = det_frames[frame_index_to_render]
    
    # Get speed threshold from params passed from the main page
    params = results.get('params', {})
    speed_threshold = params.get('moving_speed_thresh', 3.0)
    
    obj_x, obj_y, obj_z, obj_text = [], [], [], []
    for d in current_dets:
        obj_x.append(d['cx'])
        obj_y.append(d['cy'])
        obj_z.append(-6.5)
        
        speed = d.get('speed', 0.0)
        # Apply threshold: if speed is below, show only ID
        if speed >= speed_threshold:
            obj_text.append(f"ID: {d['tid']}<br>{speed:.1f} m/s")
        else:
            obj_text.append(f"ID: {d['tid']}")

    fig.add_trace(go.Scatter3d(
        x=obj_x, y=obj_y, z=obj_z,
        mode='markers+text', # Show markers and text permanently
        name='Objects',
        marker=dict(size=8, color='red', symbol='circle'),
        text=obj_text,
        textposition='top center',
        textfont=dict(size=10, color='black'),
        hoverinfo='none' # Disable hover text since it's now permanent
    ))

    # 4. Add Trajectories
    moving_tids = {d['tid'] for d in current_dets if d.get('moving', False)}
    for tid in sorted(list(moving_tids)):
        traj_x, traj_y, traj_z = [], [], []
        for i in range(frame_index_to_render + 1):
            for d in det_frames[i]:
                if d['tid'] == tid:
                    traj_x.append(d['cx']); traj_y.append(d['cy']); traj_z.append(-6.5)
                    break
        if len(traj_x) >= 2:
            fig.add_trace(go.Scatter3d(
                x=traj_x, y=traj_y, z=traj_z,
                mode='lines', name=f'Track {tid}',
                line=dict(color='magenta', width=3),
                showlegend=False # Hide trajectory legends
            ))

    # DYNAMIC AXIS RANGE based on Road Polygon
    road_poly = results['road_poly']
    minx, miny, maxx, maxy = road_poly.bounds
    
    # Add some buffer to the bounds
    buffer_x = 5.0
    buffer_y = 10.0
    
    layout_dict = dict(
        margin=dict(l=0, r=0, b=0, t=40),
        title=f"Frame {frame_index_to_render}",
        scene=dict(
            xaxis=dict(title='X (m)', range=[minx - buffer_x, maxx + buffer_x]),
            yaxis=dict(title='Y (m)', range=[miny - buffer_y, maxy + buffer_y]),
            zaxis=dict(title='Z (m)', range=[-15, 10]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.15)
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # Use camera eye from results (UI settings) if available
    camera_eye = results.get('camera_eye', {'x': 1.25, 'y': 1.25, 'z': 1.25})
    
    layout_dict['scene']['camera'] = {
        'up': {'x': 0, 'y': 0, 'z': 1},
        'center': {'x': 0, 'y': 0, 'z': 0},
        'eye': camera_eye
    }

    fig.update_layout(**layout_dict)
    return fig


import imageio
import tempfile

def generate_tracking_animation(results, output_gif_path, progress_callback=None, max_frames=0, save_frames=True):
    """
    Generates a GIF animation and optionally saves individual frames as images.
    """
    output_dir = os.path.dirname(output_gif_path)
    vis_dir = os.path.join(output_dir, "vis")
    if save_frames:
        os.makedirs(vis_dir, exist_ok=True)

    # Use the camera eye specified in the results (from UI)
    camera_eye = results.get('camera_eye', {'x': 1.25, 'y': 1.25, 'z': 1.25})
    fixed_camera = {
        'up': {'x': 0, 'y': 0, 'z': 1},
        'center': {'x': 0, 'y': 0, 'z': 0},
        'eye': camera_eye
    }

    # Create a temporary directory to store frame images
    with tempfile.TemporaryDirectory() as temp_dir:
        image_files = []
        all_frames = len(results['det_frames'])
        
        # Determine the number of frames to process
        if max_frames > 0:
            num_frames = min(max_frames, all_frames)
        else:
            num_frames = all_frames

        for i in range(num_frames):
            # Generate the figure for the current frame with the fixed UI camera
            original_pcd_path = results['original_pcd_files'][i]
            fig = create_3d_figure(results, i, original_pcd_path)
            
            # Since create_3d_figure already uses results['camera_eye'], 
            # the camera is already "fixed" to the UI values.
            # We also ensure the axis ranges are fixed in create_3d_figure.
            
            # Save the figure to a static image file
            frame_filename = f"{i:04d}.png"
            temp_frame_path = os.path.join(temp_dir, frame_filename)
            fig.write_image(temp_frame_path, width=1200, height=800, scale=1)
            image_files.append(temp_frame_path)

            # Also save to the persistent vis directory if requested
            if save_frames:
                persistent_frame_path = os.path.join(vis_dir, frame_filename)
                import shutil
                shutil.copy(temp_frame_path, persistent_frame_path)

            # Update progress if a callback is provided
            if progress_callback:
                progress_callback(i + 1, num_frames)

        # Create the GIF from the generated images
        with imageio.get_writer(output_gif_path, mode='I', duration=0.1, loop=0) as writer:
            for image_file in image_files:
                image = imageio.imread(image_file)
                writer.append_data(image)
    
    return output_gif_path

