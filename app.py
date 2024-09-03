import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO    

model = YOLO('best.pt')

# Function to perform inference and draw bounding boxes for traffic signs
def detect_objects(image):
    # Convert image to numpy array
    image_np = np.array(image)

    # Perform prediction
    results = model(image_np)

    # Initialize lists for boxes and classes
    boxes, classes = [], []

    # Process the results
    for result in results:
        for box, score, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            # Assuming class IDs correspond to traffic signs; update as necessary
            if class_id in range(0, 10):  # Adjust this range based on your model's class IDs for signs
                boxes.append(box)
                classes.append(f"Sign {int(class_id)}")  # Label based on class_id

    return boxes, classes

# Function to draw bounding boxes on the image
def draw_boxes(image, boxes, classes):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, "Sign", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

# Main app to connect the YOLO model
def main():
    st.title("Traffic Sign Detection")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Detect traffic signs
        boxes, classes = detect_objects(image)

        if boxes and classes:
            # Draw boxes on the image
            image_with_boxes = draw_boxes(np.array(image), boxes, classes)
            st.image(image_with_boxes, caption=' detected signs', use_column_width=True)
        else:
            st.warning("No traffic signs detected.")

if __name__ == "__main__":
    main()
