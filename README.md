# Assistive-AI-for-Visually-Impaired
AN assistive AI application for visually impaired individuals

# Assistive AI for Visually Impaired Individuals

This project is an assistive application designed to help visually impaired individuals understand their environment through image analysis, actionable insights, and audio guidance. The application provides scene understanding, object detection, and safety tips in a simple and accessible format.

---

## Features

1. **Scene Understanding**:
   - Generates a descriptive caption for an uploaded image using Vision Transformer (ViT) and GPT2.
   - Example: *"Cars are driving down the street."*

2. **Object Detection**:
   - Identifies objects in an image using YOLOv3.
   - Highlights objects with bounding boxes and class labels.

3. **Assistive Measures**:
   - Generates actionable safety tips based on the scene description using GPT models.
   - Converts text-based tips into audio for easy accessibility.

---

## Technologies Used

- **Frameworks**:
  - Streamlit for building the user interface.
- **Image Processing**:
  - VisionEncoderDecoderModel for scene understanding.
  - YOLOv3 for object detection.
- **Natural Language Processing**:
  - `EleutherAI/gpt-neo-125M` for generating actionable insights.
- **Audio Synthesis**:
  - Google TTS (gTTS) for converting text into speech.
- **Libraries**:
  - PyTorch
  - OpenCV
  - NumPy
  - PIL
  - Transformers

---

## Installation

### Prerequisites
1. Python 3.8 or above.
2. Install required Python libraries:
   ```bash
   pip install torch transformers pillow gtts streamlit opencv-python-headless
   ```

3. Download YOLOv3 weights and configuration files:
   - [YOLOv3 weights](https://pjreddie.com/media/files/yolov3.weights)
   - [YOLOv3 configuration](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
   - [COCO Names file](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

4. Ensure you have GPU support for PyTorch for optimal performance.

---

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/assistive-ai.git
   cd assistive-ai
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Upload an image and select a feature:
   - **Scene Understanding**: Displays a descriptive caption.
   - **Object Detection**: Displays the image with detected objects.
   - **Assistive Measures**: Generates safety tips and provides audio playback.

---

## Directory Structure

```plaintext
assistive-ai/
├── app.py                # Main Streamlit application
├── yolov3.cfg            # YOLOv3 configuration file
├── yolov3.weights        # YOLOv3 weights
├── coco.names            # COCO class labels
├── requirements.txt      # Required Python packages
└── README.md             # Project documentation
```

---

## Example Outputs

### Scene Understanding
- **Input**: Image of a street.
- **Output**: "Cars are driving down the street."

### Object Detection
- **Input**: Image of a room.
- **Output**: Objects such as "chair", "table" with bounding boxes.

### Assistive Measures
- **Input**: Image of a busy street.
- **Output**: "Stay on the sidewalk and listen for approaching vehicles." (Text and Audio)

---

## Future Enhancements

1. Real-time video processing for dynamic environments.
2. Mobile app deployment for better portability.
3. Multilingual support for wider accessibility.

---

## Contributions

Contributions are welcome! Feel free to fork the repository and submit pull requests.

---



