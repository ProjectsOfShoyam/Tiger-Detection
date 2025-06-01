# 🐅 Tiger Detection System

A professional real-time tiger detection application powered by YOLOv8 and Streamlit, designed for wildlife conservation and research purposes.


## 🌟 Features

- **Real-time Tiger Detection**: Advanced YOLOv8 model for accurate tiger identification
- **Interactive Web Interface**: Professional Streamlit-based UI with modern design
- **Live Statistics**: Real-time detection counts, confidence scores, and processing metrics
- **Configurable Settings**: Adjustable confidence thresholds and detection parameters
- **Multiple Video Formats**: Support for MP4, AVI, MOV, MKV, and WebM files
- **Progress Tracking**: Visual progress bars and frame-by-frame processing status
- **Export Ready**: Detection results suitable for research and conservation analysis

## 🎯 Use Cases

- **Wildlife Conservation**: Monitor tiger populations in protected areas
- **Research Projects**: Analyze tiger behavior and habitat usage patterns
- **Camera Trap Analysis**: Process footage from remote wildlife cameras
- **Educational Purposes**: Demonstrate AI applications in conservation biology
- **Field Research**: Real-time analysis of tiger sightings and movements

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8 or higher
pip package manager
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/ProjectsOfShoyam/Tiger-Detection
cd Tiger-Detection
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Add the trained model**
   - Place the `best_tiger.pt` model file in the project root directory
   - This model was trained using the workflow described in the "Model Training" section below

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser**
   - Navigate to `http://localhost:8501`
   - Upload a video file and start detecting tigers!

## 📋 Requirements

```txt
streamlit>=1.28.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=9.5.0
```

## 🛠️ Usage

1. **Launch the Application**: Run the Streamlit app using the command above
2. **Configure Settings**: Adjust confidence threshold in the sidebar (default: 0.5)
3. **Upload Video**: Choose a video file from supported formats
4. **Start Detection**: Click "Start Tiger Detection" to begin processing
5. **Monitor Progress**: Watch real-time detection results and statistics
6. **Review Results**: Analyze detection counts, confidence scores, and processing metrics

## 📊 Technical Details

### Model Architecture
- **Framework**: YOLOv8 (You Only Look Once v8)
- **Task**: Object Detection
- **Target Class**: Tigers (Panthera tigris)
- **Training Platform**: Google Colab with GPU acceleration
- **Dataset Tool**: Roboflow for annotation and preprocessing
- **Model File**: `best_tiger.pt` (trained custom model)
- **Input**: RGB video frames
- **Output**: Bounding boxes with confidence scores

## 🤖 Complete Development Pipeline

This project demonstrates a full end-to-end machine learning workflow using modern, accessible tools:

### 🏷️ **Data Labeling with Roboflow**
- **Platform**: [Roboflow](https://roboflow.com/) - Industry-standard computer vision platform
- **Process**:
  - Collected diverse tiger images from various sources and angles
  - Used Roboflow's intuitive web-based annotation interface
  - Drew precise bounding boxes around tigers in each image
  - Applied smart annotation features like auto-labeling suggestions
- **Data Augmentation**: Leveraged Roboflow's built-in augmentation pipeline
  - Rotation, flip, brightness/contrast adjustments
  - Noise injection and blur effects for robustness
  - Automatic train/validation/test dataset splitting (70/20/10)
- **Quality Assurance**: Used Roboflow's dataset health check tools
- **Export**: Dataset exported in YOLOv8 PyTorch format with auto-generated YAML config

### 🚀 **Model Training with Google Colab**
- **Platform**: [Google Colab](https://colab.research.google.com/) - Free GPU-accelerated Jupyter notebooks


### 🌐 **Web Application with Streamlit**
- **Framework**: [Streamlit](https://streamlit.io/) - Python web app framework
- **Development Benefits**:
  - Rapid prototyping from Python script to web app
  - Built-in widgets for file upload and real-time display
  - Easy deployment and sharing capabilities
- **Integration**: Seamless YOLOv8 model loading and inference
- **Features**: Real-time video processing with professional UI

### 📈 **Why This Stack Works**
- **Cost-Effective**: All tools used are free or have generous free tiers
- **Beginner-Friendly**: No complex setup or infrastructure management
- **Scalable**: Easy to upgrade to paid plans for larger projects
- **Professional Results**: Production-quality model and application


## 🤝 Contributing

We welcome contributions to improve the Tiger Detection System! Here's how you can help:

### Ways to Contribute
- **Bug Reports**: Submit issues for any bugs you encounter
- **Feature Requests**: Suggest new features or improvements
- **Code Contributions**: Submit pull requests with enhancements
- **Documentation**: Improve or expand project documentation
- **Dataset Improvement**: Share additional tiger images or better annotations
- **Model Optimization**: Experiment with different YOLOv8 variants or training parameters


### Future Enhancements
- [ ] **Multi-species Detection**: Extend to other wildlife species
- [ ] **Temporal Tracking**: Track individual tigers across frames
- [ ] **Geographic Integration**: GPS coordinate mapping
- [ ] **Alert System**: Real-time notifications for detections
- [ ] **Data Visualization**: Charts and graphs for analysis

## 🌍 Environmental Impact

This project contributes to wildlife conservation by:
- **Reducing Human Intervention**: Automated monitoring reduces disturbance to wildlife
- **Efficient Resource Usage**: AI-powered analysis is more efficient than manual review
- **Data-Driven Conservation**: Provides quantitative data for conservation decisions
- **Research Acceleration**: Speeds up wildlife research and population studies



## 🙏 Acknowledgments

- **Ultralytics**: For the excellent YOLOv8 framework
- **Streamlit**: For the amazing web app framework
- **Conservation Community**: For inspiring this wildlife-focused application
- **Open Source Contributors**: For continuous improvements and feedback

## 📞 Support


### Contact Information
- **Project Maintainer**: Shoyam Chaulagain (mailto : hello.shoyam@gmail.com)
- **GitHub Issues**: [Report bugs here](https://github.com/ProjectsOfShoyam/Tiger-Detection/issues)

---

<div align="center">

**🐅 Protecting Tigers Through Technology 🐅**

*Built with ❤️ for wildlife conservation*

[⭐ Star this repo](https://github.com/ProjectsOfShoyam/Tiger-Detection) | [🐛 Report Bug](https://github.com/ProjectsOfShoyam/Tiger-Detection/issues) | [💡 Request Feature](https://github.com/ProjectsOfShoyam/Tiger-Detection/issues)

</div>
