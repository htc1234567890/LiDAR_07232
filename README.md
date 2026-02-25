# Lidar Processing Toolkit 🚗

This is a 3D Lidar point cloud processing toolkit developed using **Streamlit**. It is designed for point cloud visualization, background filtering, and object detection and tracking.

## Core Modules
1.  **Background Filtering**: Build a background model to filter out static objects like the ground and walls.
2.  **Object Detection and Tracking**: Identify and continuously track dynamic objects (such as cars and trucks) in the filtered point cloud.

---

## Installation Guide

If you are installing this project on a brand new Windows computer, please follow these steps:

### 1. Install Python
*   Go to the [Python Official Website](https://www.python.org/downloads/windows/) and download **Python 3.12**.
*   **Important**: Make sure to check the box **"Add Python to PATH"** during installation.

### 2. Get the Code
*   Download the project ZIP file from GitHub and extract it, or use Git to clone:
    ```bash
    git clone https://github.com/htc1234567890/LiDAR_07232.git
    cd LiDAR_07232
    ```

### 3. Install Dependencies
*   Open PowerShell or Command Prompt (CMD) and navigate to the project root directory.
*   Run the following command to install all necessary libraries:
    ```bash
    pip install -r requirements.txt
    ```

### 4. Prepare Data
*   Place your `.pcd` point cloud files into the `data/` folder in the project root.

### 5. Run the Application
*   Type the following command in the terminal to start the web interface:
    ```bash
    streamlit run Home.py
    ```
*   The application will automatically open in your default browser.

---

## File Structure
*   `Home.py`: Entry point for the Streamlit application.
*   `pages/`: Contains specific pages for background filtering and object detection.
*   `bg_filter_core.py`: Core algorithm logic for background filtering.
*   `detection_logic.py`: Core algorithm logic for object detection and tracking.
*   `data/`: Directory for input `.pcd` files (ignored by .gitignore, needs manual input).
*   `outputs/`: Directory for processing result outputs.
