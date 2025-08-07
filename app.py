import torch
import torch.nn as nn
import streamlit as st
import numpy as np
import cv2
from ultralytics import YOLO
from transformer_model import ConfidenceTransformer
from gesture_model import GestureClassifier

# ----------------------------
# Vocabulary and Token Mapping
# ----------------------------
vocab = [
    "i", "want", "to", "drink", "eat", "stop", "help", "need", "can", "you",
    "me", "please", "<mask>", "<pad>", "pain", "toilet", "hello",
    "thank", "thank you", "yes", "no", "feel", "say", "the", "wants", "hi"
]
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for i, word in enumerate(vocab)}
mask_token_id = word2idx["<mask>"]
gesture_labels = ['drink', 'eat', 'pain', 'toilet', 'hello', 'thank you', 'help', 'yes', 'no', 'stop']

# ----------------------------
# Load Models
# ----------------------------
gesture_model = GestureClassifier(num_classes=10)
gesture_model.load_state_dict(torch.load("gesture_model.pth", map_location=torch.device("cpu")))
gesture_model.eval()

transformer_model = ConfidenceTransformer(vocab_size=len(vocab), embed_dim=64, output_dim=len(vocab))
transformer_model.load_state_dict(torch.load("confidence_transformer.pth", map_location=torch.device("cpu")))
transformer_model.eval()

# ----------------------------
# Streamlit UI Config
# ----------------------------
st.set_page_config(page_title="Multimodal Word Prediction", layout="centered")
st.title("🧠 Transformer-Based Multimodal Word Prediction App")
st.write("Predict masked words using sentence, confidence score, and gestures.")

# ----------------------------
# Step 1: Sentence + Confidence Input
# ----------------------------
st.header("Step 1: Enter Sentence and Confidence")
sentence = st.text_input("Enter a sentence with <MASK>", "I feel <MASK>")
confidence = st.slider("Confidence score (0.0 to 1.0)", 0.0, 1.0, 0.9)

# Utility to tokenize sentence
def tokenize_sentence(sentence):
    words = sentence.lower().replace(".", "").split()
    return [word2idx.get(w, word2idx["<pad>"]) for w in words]

# ----------------------------
# Step 2: Lexical Prediction
# ----------------------------
st.subheader("Step 2: Predict <MASK> Word from Sentence")
if st.button("📖 Predict using Sentence + Confidence Only"):
    if "<mask>" not in sentence.lower():
        st.warning("⚠️ Please include '<MASK>' in your sentence.")
    else:
        token_ids = tokenize_sentence(sentence)
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        conf_tensor = torch.tensor([[confidence]], dtype=torch.float32)

        with torch.no_grad():
            output_logits = transformer_model(input_ids, conf_score=conf_tensor)
            pred_index = torch.argmax(output_logits).item()
            lexical_prediction = idx2word[pred_index]

        st.success(f"📝 Lexical Prediction: **{lexical_prediction}**")
        st.markdown(f"📈 Type: **{'normal' if confidence >= 0.5 else 'aphasic'}**, Confidence Score: `{confidence}`")

# ----------------------------
# Step 3: Gesture Input (YOLO)
# ----------------------------
st.header("Step 3: Provide Gesture Input")
use_webcam = st.checkbox("Use Webcam for Gesture Capture")

# Initialize session state to store predicted gesture
if 'predicted_gesture' not in st.session_state:
    st.session_state['predicted_gesture'] = None

if use_webcam:
    if st.button("🎥 Start Webcam"):
        model = YOLO("yolov8n-pose.pt")
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        frames = []

        for _ in range(30):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", caption="Live Feed", use_container_width=True)
            results = model(frame, verbose=False)
            keypoints = results[0].keypoints
            if keypoints is not None and len(keypoints) > 0:
                kp_array = keypoints.xy[0].cpu().numpy().flatten()
                if len(kp_array) == 34:
                    frames.append(kp_array)

        cap.release()
        if frames:
            keypoints_2d = np.array(frames).T  # shape: [34, sequence_len]
            gesture_tensor = torch.tensor(keypoints_2d, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                gesture_logits = gesture_model(gesture_tensor)
                gesture_index = torch.argmax(gesture_logits).item()
                st.session_state['predicted_gesture'] = gesture_labels[gesture_index]
            st.success(f"🎯 Detected Gesture: **{st.session_state['predicted_gesture']}**")
        else:
            st.error("❌ No valid keypoints detected.")

# ----------------------------
st.header("Step 4: Predict Using Sentence + Gesture")
if st.button("🔮 Predict Final Word (Multimodal)"):
    if not st.session_state.get('predicted_gesture'):
        st.warning("⚠️ Please capture a gesture first.")
    else:
        token_ids = tokenize_sentence(sentence)
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        conf_tensor = torch.tensor([[confidence]], dtype=torch.float32)

        with torch.no_grad():
            output_logits = transformer_model(input_ids, conf_score=conf_tensor)
            pred_index = torch.argmax(output_logits).item()
            final_word = idx2word[pred_index]

        st.success(f"✅ Final Predicted Word: **{final_word}**")
        st.markdown(f"🧾 Sentence: `{sentence}`")
        st.markdown(f"🧠 Gesture: **{st.session_state['predicted_gesture']}**")
        st.markdown(f"📈 Type: **{'normal' if confidence >= 0.5 else 'aphasic'}**, Score: `{confidence}`")

        # Final sentence with prediction
        final_sentence = sentence.replace("<MASK>", final_word)
        st.markdown(f"📝 **Final Sentence:** `{final_sentence}`")



