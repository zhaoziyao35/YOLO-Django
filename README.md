# 收费站视频分析系统

基于 Django + YOLOv8 的收费站视频分析系统，支持：
- 拥堵检测
- 行人闯入检测
- 非机动车闯入检测

## 快速启动

### 1. 安装依赖
```bash
pip install -r requirements.txt
## 启动服务
```bash
pip install -r requirements.txt
python manage.py runserver 9060

#项目结构
tollgate-analyzer/
├── tollgate_project/ # Django 项目配置
├── traffic_api/ # 核心应用
│ ├── views.py # API 接口
│ ├── engine.py # YOLO 推理引擎
│ ├── utils.py # 工具函数
│ └── models.py # 任务模型
├── yolov8s.pt # YOLOv8 模型文件
├── test_api.py # 测试脚本
└── requirements.txt # 依赖列表


## 配置说明

编辑 `tollgate_project/settings.py`：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `DATA_UPLOAD_MAX_MEMORY_SIZE` | 50MB | 上传文件大小上限 |
| `FILE_UPLOAD_MAX_MEMORY_SIZE` | 50MB | 单文件最大内存大小 |




## 检测逻辑

1. 图片送入 YOLOv8 模型推理
2. 识别行人、车辆、自行车、摩托车等目标
3. 统计各类目标数量
4. 触发事件判断：
   - 车辆 > 5 → 拥堵
   - 行人进入 y < 200 区域 → 行人闯入
   - 检测到自行车/摩托车 → 非机动车闯入


   ## 测试结果示例

运行 `python test_api.py` 后输出：

```json
{
  "statistics": {
    "person_count": 4,
    "vehicle_count": 6
  },
  "warning_types": ["拥堵", "行人闯入"]
}