# Drominator

FPV drone autonomous vision system using YOLOv11n.

## Requirements

- Linux
- NVIDIA GPU with CUDA 13+
- Python 3.12+
- OpenCV (for camera capture)
- CMake 3.10+

## Installation

### 1. Clone the repo

```bash
git clone <repo-url>
cd Drominator
```

### 2. Set up Python environment

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip ultralytics onnx onnxslim onnxruntime-gpu
```

### 3. Download the AI model

```bash
.venv/bin/python3 -c "
from ultralytics import YOLO
import shutil, os
os.makedirs('models', exist_ok=True)
model = YOLO('yolo11n.pt')
model.export(format='onnx', imgsz=320, simplify=True)
shutil.move('models/yolo11n.onnx', 'models/yolo11n_320.onnx')
print('Done -> models/yolo11n_320.onnx')
"
```

### 4. Build the C++ camera module

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### 5. Run

```bash
./build/minicam
```

Press `Esc` to quit.

## Camera

Default device: `/dev/video4` (Macrosilicon EWRF receiver).
Change `camera_index` in [main.cpp](main.cpp) if needed.
