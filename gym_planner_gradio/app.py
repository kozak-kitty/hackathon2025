import gradio as gr
from ultralytics import YOLO
from PIL import Image

# Load your trained YOLOv8 model (relative path)
model = YOLO("../best.pt")

# Equipment class names from your dataset
CLASS_NAMES = [
    'Chest Press machine',
    'Lat Pull Down',
    'Seated Cable Rows',
    'arm curl machine',
    'chest fly machine',
    'chinning dipping',
    'lateral raises machine',
    'leg extension',
    'leg press',
    'reg curl machine',
    'seated dip machine',
    'shoulder press machine',
    'smith machine'
]

def detect_equipment(image):
    # Run YOLO inference
    results = model(image)

    # Get detection results
    detections = results[0].boxes
    annotated_image = results[0].plot()  # Draw bounding boxes

    detected_classes = []
    if detections is not None and len(detections) > 0:
        for box in detections:
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"Unknown ({cls_id})"
            detected_classes.append(f"{label} ({conf:.2f})")
    else:
        detected_classes.append("No equipment detected.")

    # Convert to PIL for Gradio
    annotated_pil = Image.fromarray(annotated_image)

    return annotated_pil, "\n".join(detected_classes)


# Build Gradio interface
interface = gr.Interface(
    fn=detect_equipment,
    inputs=gr.Image(type="filepath", label="Upload a gym photo"),
    outputs=[
        gr.Image(label="Detected Equipment"),
        gr.Textbox(label="Detected Classes")
    ],
    title="Gym Equipment Detection",
    description="Upload an image of gym equipment to detect its type using your custom YOLOv8 model."
)

if __name__ == "__main__":
    interface.launch(debug=True)
