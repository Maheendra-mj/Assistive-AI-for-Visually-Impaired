import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
import pytesseract
from gtts import gTTS
from PIL import Image
import cv2
import numpy as np
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

# Function for Scene Understanding
def scene_understanding(image):
    """
    Generates a descriptive caption for the given image using Google's ViT-GPT2 model.
    """
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    inputs = processor(images=image, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        generated_ids = model.generate(inputs['pixel_values'], max_length=50, num_beams=5)
    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return caption

# Function for Text-to-Speech
def text_to_speech(image):
    description = scene_understanding(image)
    if description:
        speech = gTTS(description)
        speech.save("scene_description.mp3")
        return description, "scene_description.mp3"
    return "No description could be generated.", None

# Function for Object Detection
def main_object_detection(image):
    image_np = np.array(image)
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    with open("coco.names", "r") as f:
        classes = f.read().strip().split("\n")
    height, width, _ = image_np.shape
    blob = cv2.dnn.blobFromImage(image_np, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y, box_width, box_height = (obj[0:4] * [width, height, width, height]).astype("int")
                x = int(center_x - box_width / 2)
                y = int(center_y - box_height / 2)
                boxes.append([x, y, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        label = f"{classes[class_ids[i]]} ({confidences[i]:.2f})"
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image_np, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return Image.fromarray(image_np)

# Streamlit App
st.set_page_config(page_title="Assistive AI for Visually Impaired", layout="wide")

st.sidebar.title("Assistive AI")
st.sidebar.write("Choose a feature to explore:")

# Sidebar Navigation
feature = st.sidebar.radio("Select Feature", ["Scene Understanding", "Text-to-Speech", "Object Detection"])

# File Upload
uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Main Area
st.title("Assistive AI for Visually Impaired")
st.write("This application provides assistance by analyzing images and performing tasks like Scene Understanding, Text-to-Speech conversion, and Object Detection.")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated parameter

    if feature == "Scene Understanding":
        st.subheader("Scene Understanding")
        st.write("Generating a descriptive caption for the uploaded image...")
        description = scene_understanding(image)
        st.success(f"Scene Description: {description}")

    elif feature == "Text-to-Speech":
        st.subheader("Text-to-Speech")
        st.write("Converting the scene description into audio...")
        text, audio_file = text_to_speech(image)
        if audio_file:
            st.success(f"Scene Description: {text}")
            st.audio(audio_file)
        else:
            st.error("Could not generate an audio file.")

    elif feature == "Object Detection":
        st.subheader("Object Detection")
        st.write("Detecting objects in the uploaded image...")
        detected_image = main_object_detection(image)
        st.image(detected_image, caption="Objects Detected", use_container_width=True)  # Updated parameter

