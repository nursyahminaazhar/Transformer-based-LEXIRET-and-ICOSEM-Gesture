# Transformer-based-LEXIRET-and-ICOSEM-Gesture
User Interface - Transformer-based Lexical Retrieval and Iconic and Semantic Gesture

# 🧠 Multimodal Transformer UI for Lexical Retrieval and Gesture Understanding

This is a web-based **multimodal application** that combines **Transformer-based lexical retrieval** with **iconic and semantic gesture recognition**, designed to assist normal and aphasic individuals in word prediction tasks.

Built using **Streamlit**, this application integrates two models:
- A **Transformer-based lexical model** (`confidence_transformer.pth`)
- A **gesture-based CNN model** (`gesture_model.pth`) trained on YOLOv8-pose keypoints

---

## 📁 Folder Structure
multimodal_app/
│
├── app.py # Main Streamlit app
├── confidence_transformer.pth # Lexical Transformer model
├── gesture_model.pth # Trained gesture classification model
├── gesture_model.py # CNN model architecture for gesture
├── transformer_model.py # Transformer model architecture
├── yolov[n]-pose.pt # YOLOv8 pose model (for keypoint extraction)
├── style/ # Optional folder for UI styling (CSS/images)

How to Run the Web App

### 1. Prerequisites

Ensure you have Python installed and the following dependencies:

```bash
pip install torch torchvision
pip install streamlit
pip install opencv-python
pip install yolov5  # or ultralytics, if you're using YOLOv8

Run the App
Open terminal and run the following:
bash: cd path\to\multimodal_app
streamlit run app.py

Default browser: http://localhost:8501

