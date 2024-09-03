from ultralytics import YOLO

# Step 1: Load the pre-trained YOLOv8n model
model = YOLO("yolov8n.yaml")  # build a new model from YAML
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Step 2: Train the model on your custom dataset
model.train(
    data='data.yaml',  # 指定你的data.yaml文件路径
    epochs=50,         # 设定训练的轮数，你可以根据需要调整
    batch=16,     # 批处理大小，视你的GPU内存大小而定
    imgsz=640,         # 输入图像的尺寸，YOLOv8默认是640x640
    name='yolov8n-honeybee',  # 训练运行的名称
    verbose=True,       # 显示详细的训练过程
    device='cuda'
)
