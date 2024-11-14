from ultralytics import YOLO
import gradio as gr
from gradio_toggle import Toggle
import cv2
import numpy as np
import time


import onnxruntime as ort # For ONNX
import tensorrt as trt # For TensorRT
import openvino.runtime as ov  # For OpenVINO

def initialize_models():
    """Initialize models at startup"""

    
    print("Loading detection model...")
    det_model = YOLO("C:/Users/Christian/Desktop/best_det/best.pt", task="detect").to('cuda')
    # det_model = YOLO("C:/Users/Christian/Desktop/best_det/best.engine", task="detect")
    # det_model = YOLO("C:/Users/Christian/Desktop/best_det/best_openvino/best_openvino_model/", task="detect")

    print("Loading segmentation model...")
    seg_model = YOLO("C:/Users/Christian/Desktop/best_seg/best.pt", task="segment").to('cuda')
    # seg_model = YOLO("C:/Users/Christian/Desktop/best_seg/best.engine", task="segment")
    # seg_model = YOLO("C:/Users/Christian/Desktop/best_seg/best_openvino/best_openvino_model/", task="segment")
       
    return det_model, seg_model

# Initialize models globally before Gradio interface
model_det, model_seg = initialize_models()


def process_frame(frame, is_segmentation):
    start_time = time.time()
    
    # Use detection when toggle is off, segmentation when on
    model = model_seg if is_segmentation else model_det
    
    results = model.predict(
        source=frame,
        conf=0.6,
        line_width=2,
        save_crop=False,
        save_txt=False,
        show_labels=True,
        show_conf=True,
        verbose=False, 
        device="cuda",#"cpu", #0,
        half=True, #True
    )
    
    annotated_img = results[0].plot()
    
    # Only calculate metrics if there are detections
    if len(results[0].boxes) > 0:
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time
        fps_text = f"FPS: {fps:.1f}"
        inf_text = f"Inference: {inference_time*1000:.1f}ms"
    else:
        fps_text = "FPS: 0.0"
        inf_text = "Inference: 0.0ms"
    
    # Draw FPS and inference time counters
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(annotated_img, fps_text, (annotated_img.shape[1]-180, 30), 
                font, 1, (0, 255, 0), 2)
    cv2.putText(annotated_img, inf_text, (annotated_img.shape[1]-300, 60), 
                font, 1, (0, 255, 0), 2)
    
    return annotated_img

css=""".my-group {max-width: 800px !important; margin: auto !important;}
      .toggle-row {display: flex !important; justify-content: center !important; margin-bottom: 20px !important;}
      .container {display: flex !important; flex-direction: column !important; align-items: center !important;}"""

with gr.Blocks(css=css) as demo:
    with gr.Group(elem_classes="my-group"):
        with gr.Row(elem_classes="toggle-row"):
            mode_toggle = Toggle(
                label="Switch to Segmentation",
                value=False,
                interactive=True,
                color="green",
                info="Switch between object detection and segmentation (Default: Detection)"
            )
        input_img = gr.Image(sources=['webcam'], type="numpy", streaming=True, label="Webcam")
        input_img.stream(
            fn=process_frame,
            inputs=[input_img, mode_toggle],
            outputs=input_img,
            stream_every=0.033  # 30 FPS
        )

demo.launch()