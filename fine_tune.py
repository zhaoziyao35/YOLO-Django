from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8s.pt')

# 开始微调
results = model.train(
    data='tollgate-dataset/data.yaml',  # 数据集配置文件
    epochs=50,                          # 训练轮数
    imgsz=640,                          # 输入图片尺寸
    batch=16,                           # 批次大小（显存不够就改小，如8或4）
    device='cpu',                       # 如果有GPU，改成 'cuda'
    workers=4,
    lr0=0.001,
    freeze=10,
    name='traffic_police_finetune',
    exist_ok=True,
)

print("✅ 微调完成！")
print(f"最佳模型: {results.save_dir}/weights/best.pt")