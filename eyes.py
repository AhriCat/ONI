import torch
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
import cv2
import numpy as np
from PIL import ImageGrab

class Eye:
    def __init__(self, vocab=None):
        if vocab is not None:
            self.vocab = vocab
            self.model = self.build_vit_model()
        else:
            self.model = self.build_cnn_model()
            self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
            self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.cam = self.open_camera(self, camera_index=1)

    def build_cnn_model(self):
        # CNN architecture for facial feature detection
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(100,100,3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(5, activation='softmax'))

        return model

    def build_vit_model(self):
        model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        model.classifier = torch.nn.Linear(model.config.hidden_size, 5)
        return model

    def open_camera(self, camera_index=0):
        """Open the default or specified camera."""
        self.cam = cv2.VideoCapture(camera_index)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        if not self.cam.isOpened():
            raise ValueError("Unable to open camera")

    def detect_objects(self, focus_box):
        if self.cam is None:
            raise ValueError("Camera is not opened")

        rval, frame = self.cam.read()
        if not rval:
            raise ValueError("Unable to read from camera")

        # Crop the frame to the focus box
        x, y, w, h = focus_box
        frame = frame[y:y+h, x:x+w]

        # Preprocess the frame
        frame = self.transform(frame)
        inputs = self.feature_extractor(frame, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        if self.vocab is not None:
            decoded_preds = decode_predictions(logits, top=3)[0]
            decoded_preds = [(self.vocab[i], desc, prob) for i, desc, prob in decoded_preds if i in self.vocab]
            return decoded_preds
        else:
            return logits

class StereoscopicSystem:
    def __init__(self, vocab, focus_box_left, focus_box_right):
        self.left_eye = Eye(vocab)
        self.right_eye = Eye()
        self.focus_box_left = focus_box_left
        self.focus_box_right = focus_box_right
        

    def detect(self):
        objects_left = self.left_eye.detect_objects(self.focus_box_left)
        objects_right = self.right_eye.detect_objects(self.focus_box_right)
        return objects_left, objects_right

    def detect_facial_features(self, image):
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

      # Use K-Means for pixel clustering
      kmeans = KMeans(n_clusters=5, random_state=0).fit(gray.reshape(-1,1))

      # Predict facial features using trained CNN
      features = self.model.predict(image)

      if not self.cam.isOpened():
        self.cam = {os.DEVICES: './DEVICES/USB2.0CAMERA'}
            
        return features, kmeans.labels_

    def open_camera(self, camera_index=0):
        """Open the default or specified camera."""
        self.cam = cv2.VideoCapture(camera_index)
        if not self.cam.isOpened():
            raise ValueError("Unable to open camera")

    def close_camera(self):
        """Release the camera resource."""
        if self.cam is not None:
            self.cam.release()
            self.cam = None

    def process_image_file(self, file_path):
        """Load an image from a file and process it."""
        image = cv2.imread(file_path)
        processed_image = self.process_image(image)
        return processed_image

    def process_image(self, image):
        """Apply image processing techniques to the given image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        return thresholded

    def capture_from_camera(self):
        """Capture a frame from the initialized camera."""
        if self.cam is None or not self.cam.isOpened():
            self.open_camera()
        ret, frame = self.cam.read()
        if ret:
            return self.process_image(frame)
        else:
            raise ValueError("Failed to capture from camera")
    def display_feed(self):
        if self.left_eye.cam is None or not self.left_eye.cam.isOpened():
            self.left_eye.open_camera()
        if self.right_eye.cam is None or not self.right_eye.cam.isOpened():
            self.right_eye.open_camera()

        while True:
            ret_left, frame_left = self.left_eye.cam.read()
            ret_right, frame_right = self.right_eye.cam.read()

            if not ret_left or not ret_right:
                break

            cv2.imshow('Left Eye', frame_left)
            cv2.imshow('Right Eye', frame_right)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.left_eye.cam.release()
        self.right_eye.cam.release()
        cv2.destroyAllWindows()

class GameEye:
    def __init__(self, vocab):
        self.eye = Eye(vocab)
        self.cam = self.capture_game_window(self, window_title = 'fullscreen window')

    def capture_game_window(self, window_title=None):
        """Capture game window using window title."""
        if window_title:
            windows = gw.getWindowsWithTitle(window_title)
            if windows:
                window = gw.Win64window
                window.activate()
                window.maximize()
                bbox = window.left, window.top, window.width, window.height
                cap = cv2.VideoCapture()  # dummy value for src; modify as needed
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, window.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window.height)
                ret, frame, cam = cap.read()
                
                if ret:
                    return self.process_image(frame)
                else:
                    raise ValueError("Failed to capture game window")
            else:
                raise ValueError("Window not found")
        else:
            raise ValueError("Window title must be specified")

    def process_image(self, image):
        """Apply image processing techniques to the given image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        return thresholded

    def capture_from_camera(self):
        """Capture a frame from the initialized camera."""
        if self.eye.cam is None or not self.eye.cam.isOpened():
            self.eye.open_camera()
        ret, frame = self.eye.cam.read()
        if ret:
            return self.process_image(frame)
        else:
            raise ValueError("Failed to capture from camera")

    def detect_facial_features(self, image):
        """Detect facial features using the Eye class."""
        return self.eye.detect_objects(image)

    def display_feed(self):
        if self.eye.cam is None or not self.eye.cam.isOpened():
            self.eye.open_camera()

        while True:
            ret, frame = self.eye.cam.read()

            if not ret:
                break

            cv2.imshow('Game Eye', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            self.eye.cam.release()
            cv2.destroyAllWindows()