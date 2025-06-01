import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np
from pathlib import Path
import time
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Tiger Detection System",
    page_icon="🐅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b35, #ff8e53);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff6b35;
        margin: 1rem 0;
    }
    
    .stats-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .detection-frame {
        border: 3px solid #ff6b35;
        border-radius: 10px;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path="best_tiger.pt"):
    """Load and cache the YOLO model"""
    try:
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found. Please ensure the model file is in the same directory.")
            return None
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

class TigerDetector:
    def __init__(self, model):
        self.model = model
        self.detection_count = 0
        self.frame_count = 0
        self.confidence_threshold = 0.5
    
    def detect_in_frame(self, frame):
        """Detect tigers in a single frame"""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run detection
            results = self.model(frame_rgb, conf=self.confidence_threshold)
            
            # Count detections
            detections = results[0].boxes
            if detections is not None:
                tiger_count = len(detections)
                if tiger_count > 0:
                    self.detection_count += tiger_count
            
            # Annotate frame
            annotated_frame = results[0].plot()
            return annotated_frame, detections
            
        except Exception as e:
            st.error(f"Error during detection: {str(e)}")
            return frame, None
    
    def process_video(self, video_path, progress_bar, status_text):
        """Process video file for tiger detection"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            st.error("Error: Could not open video file.")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create columns for layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            stframe = st.empty()
        
        with col2:
            stats_container = st.empty()
        
        self.detection_count = 0
        self.frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Process frame
            annotated_frame, detections = self.detect_in_frame(frame)
            
            # Update display
            with stframe.container():
                st.markdown('<div class="detection-frame">', unsafe_allow_html=True)
                st.image(annotated_frame, channels="RGB", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Update statistics
            detection_rate = (self.detection_count / self.frame_count) * 100 if self.frame_count > 0 else 0
            
            with stats_container.container():
                st.markdown("""
                <div class="stats-container">
                    <h3>📊 Detection Statistics</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.metric("Total Detections", self.detection_count)
                st.metric("Frames Processed", self.frame_count)
                st.metric("Detection Rate", f"{detection_rate:.1f}%")
                
                if detections is not None and len(detections) > 0:
                    max_conf = float(detections.conf.max()) if len(detections.conf) > 0 else 0
                    st.metric("Max Confidence", f"{max_conf:.2f}")
            
            # Update progress
            progress = min(self.frame_count / total_frames, 1.0)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {self.frame_count}/{total_frames}")
            
            # Add small delay to make visualization smoother
            time.sleep(0.03)
        
        cap.release()
        status_text.success("✅ Video processing completed!")

def main():
    # Header
    st.markdown('<h1 class="main-header">🐅 Tiger Detection System</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    <div class="info-box">
        <h3>🎯 About This Application</h3>
        <p>This advanced tiger detection system uses YOLOv8 deep learning model to identify and track tigers in video footage. 
        Upload your video file and watch as the AI analyzes each frame in real-time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model loading
        with st.spinner("Loading AI model..."):
            model = load_model()
        
        if model is None:
            st.stop()
        
        st.success("✅ Model loaded successfully!")
        
        # Detection settings
        st.subheader("Detection Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        # Model info
        st.subheader("📋 Model Information")
        st.info(f"""
        **Model Type:** YOLOv8
        **Task:** Object Detection
        **Target:** Tigers
        **Confidence:** {confidence_threshold}
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Upload Video")
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help="Supported formats: MP4, AVI, MOV, MKV, WEBM"
        )
    
    with col2:
        st.subheader("💡 Tips")
        st.markdown("""
        - Use clear, high-quality videos
        - Ensure good lighting conditions
        - Videos with tigers in natural habitats work best
        - Processing time depends on video length
        """)
    
    if uploaded_video is not None:
        # Display video info
        file_details = {
            "Filename": uploaded_video.name,
            "File size": f"{uploaded_video.size / (1024*1024):.2f} MB",
            "File type": uploaded_video.type
        }
        
        st.subheader("📄 Video Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filename", file_details["Filename"])
        with col2:
            st.metric("Size", file_details["File size"])
        with col3:
            st.metric("Type", file_details["File type"])
        
        # Process video button
        if st.button("🚀 Start Tiger Detection", type="primary"):
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                    tfile.write(uploaded_video.read())
                    temp_video_path = tfile.name
                
                # Initialize detector
                detector = TigerDetector(model)
                detector.confidence_threshold = confidence_threshold
                
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                status_text.info("🔄 Initializing video processing...")
                
                # Process video
                detector.process_video(temp_video_path, progress_bar, status_text)
                
                # Cleanup
                os.unlink(temp_video_path)
                
                # Final results
                st.balloons()
                st.success(f"🎉 Detection completed! Found {detector.detection_count} tiger detections in {detector.frame_count} frames.")
                
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                st.info("Please try uploading a different video file or check the model file.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🐅 Tiger Detection System | Powered by YOLOv8 & Streamlit</p>
        <p><small>For wildlife conservation and research purposes</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()