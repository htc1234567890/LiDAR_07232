# Lidar Processing Toolkit 🚗

这是一个基于 **Streamlit** 开发的 3D 激光雷达（Lidar）点云数据处理工具包，主要用于点云的可视化、背景过滤以及目标检测与跟踪。

## 功能模块
1.  **背景过滤 (Background Filtering)**: 建立背景模型，过滤静态物体（如地面、墙壁）。
2.  **目标检测与跟踪 (Object Detection and Tracking)**: 在过滤后的点云中识别并持续跟踪动态物体（如汽车、卡车）。

---

## 新电脑安装指南 (Windows)

如果您是在一台全新的 Windows 电脑上安装此项目，请按照以下步骤操作：

### 1. 安装 Python
*   前往 [Python 官网](https://www.python.org/downloads/windows/) 下载并安装 **Python 3.12**。
*   **重要**：安装时请务必勾选 **"Add Python to PATH"**。

### 2. 获取代码
*   从 GitHub 下载此项目压缩包并解压，或者使用 Git 克隆：
    ```bash
    git clone https://github.com/htc1234567890/LiDAR_07232.git
    cd LiDAR_07232
    ```

### 3. 安装依赖环境
*   打开 PowerShell 或命令提示符 (CMD)，进入项目根目录。
*   运行以下命令安装所有必要的库：
    ```bash
    pip install -r requirements.txt
    ```

### 4. 准备数据
*   将您的 `.pcd` 点云文件放入项目根目录下的 `data/` 文件夹中。

### 5. 运行程序
*   在终端输入以下命令启动网页界面：
    ```bash
    streamlit run Home.py
    ```
*   程序会自动在浏览器中打开主页。

---

## 文件结构说明
*   `Home.py`: 应用程序入口。
*   `pages/`: 包含背景过滤和目标检测的具体功能页面。
*   `bg_filter_core.py`: 背景过滤算法核心逻辑。
*   `detection_logic.py`: 目标检测与跟踪算法核心逻辑。
*   `data/`: 用于存放输入的 `.pcd` 点云文件（已在 .gitignore 中忽略，需手动放入）。
*   `outputs/`: 处理结果的输出目录。

