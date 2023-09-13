from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import pyttsx3
import cv2

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize the model, feature extractor, and tokenizer
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        # Initialize the text-to-speech engine
        self.engine = pyttsx3.init()

        # Set up the UI
        self.setWindowTitle("Vision Beyond Sight")
        self.setGeometry(100, 100, 800, 600)

        # Create the image label
        self.image_label = QLabel(self)
        self.image_label.setGeometry(50, 50, 400, 400)
        self.image_label.setAlignment(Qt.AlignCenter)

        # Create the buttons
        self.upload_button = QPushButton("Upload Photo", self)
        self.upload_button.setGeometry(500, 50, 200, 50)
        self.upload_button.clicked.connect(self.upload_photo)

        self.capture_button = QPushButton("Capture Image", self)
        self.capture_button.setGeometry(500, 150, 200, 50)
        self.capture_button.clicked.connect(self.capture_image)

        self.generate_button = QPushButton("Generate Caption", self)
        self.generate_button.setGeometry(500, 250, 200, 50)
        self.generate_button.clicked.connect(self.generate_caption)

        self.speaker_button = QPushButton("Text-to-Speech", self)
        self.speaker_button.setGeometry(500, 350, 200, 50)
        self.speaker_button.clicked.connect(self.text_to_speech)

        # Set up the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Set up the generation parameters
        self.max_length = 16
        self.num_beams = 4
        self.gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams}

    def upload_photo(self):
        # Open a file dialog to select an image file
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Image Files (*.jpg *.jpeg *.png)")
        file_dialog.exec_()
        file_path = file_dialog.selectedFiles()[0]

        # Load the image and display it in the label
        image = QPixmap(file_path)
        self.image_label.setPixmap(image)

        # Save the image path for later use
        self.image_path = file_path

    def capture_image(self):
        # Display a message while the image is being captured
        print("Capturing image...")

        # Capture an image from the camera
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        # Convert the image to RGB and save it
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_path = "captured_image.jpg"
        image.save(image_path)

        # Display the image in the label
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap)

        # Save the image path for later use
        self.image_path = image_path

    def generate_caption(self):
        # Display a message while the image is being processed
        print("Generating caption...expected time 1m")

        # Load the image from the saved path
        image = Image.open(self.image_path)
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        # Generate the caption
        pixel_values = self.feature_extractor(images=[image], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)
        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        caption = preds[0].strip()

        # Display the caption in the label
        self.image_label.setText(caption)

        # Save the caption for later use
        self.caption = caption

    def text_to_speech(self):
        # Use the text-to-speech engine to speak the caption
        self.engine.say(self.caption)
        self.engine.runAndWait()

if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()
