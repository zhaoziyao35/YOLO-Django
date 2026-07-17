import os
import json
import logging
import time
from datetime import datetime

import numpy as np
import cv2
from django.conf import settings
from django.http import HttpResponse
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from . import utils
from .engine import get_model, start_engine, stop_engine, CURRENT_ENGINES
from .models import Task, tasks_store
from .serializers import ImageDetectSerializer, TaskCreateSerializer

logger = logging.getLogger(__name__)


@api_view(['GET'])
def health_check(request):
    return Response({'status': 'ok', 'message': '收费站视频分析服务运行中'})


@api_view(['POST'])
def detect_image(request):
    """收费站图片检测接口"""
    serializer = ImageDetectSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    image_file = serializer.validated_data['image']
    device_id = serializer.validated_data.get('device_id', 'tollgate_detect')
    conf_threshold = serializer.validated_data.get('conf_threshold', 0.25)
    model_path = serializer.validated_data.get('model_path', '')

    if not model_path:
        model_path = os.path.join(settings.BASE_DIR, 'yolov8s.pt')

    if not os.path.isfile(model_path):
        return Response({
            'success': False,
            'message': f'模型文件不存在: {model_path}'
        }, status=status.HTTP_400_BAD_REQUEST)

    # 解码图片
    file_bytes = image_file.read()
    np_arr = np.frombuffer(file_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return Response({
            'success': False,
            'message': '无效的图片文件'
        }, status=status.HTTP_400_BAD_REQUEST)

    # 加载模型并推理
    try:
        model = get_model(model_path)
    except Exception as e:
        logger.exception(f"[图片检测] 模型加载失败: {e}")
        return Response({
            'success': False,
            'message': f'模型加载失败: {e}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    results = model(frame, conf=conf_threshold, iou=0.7, verbose=False)
    detections = results[0].boxes

    # 提取检测结果
    raw_detections = []
    boxes_xyxy = np.empty((0, 4))
    cls_ids = np.empty((0,), dtype=int)
    confs = np.empty((0,))

    if detections is not None and len(detections) > 0:
        boxes_xyxy = detections.xyxy.cpu().numpy() if hasattr(detections.xyxy, 'cpu') else detections.xyxy
        cls_ids = detections.cls.cpu().numpy() if hasattr(detections.cls, 'cpu') else detections.cls
        confs = detections.conf.cpu().numpy() if hasattr(detections.conf, 'cpu') else detections.conf

        keep = utils.class_specific_nms(boxes_xyxy, confs, cls_ids, iou_thresh=0.6)
        if len(keep) < len(boxes_xyxy):
            boxes_xyxy = boxes_xyxy[keep]
            cls_ids = cls_ids[keep]
            confs = confs[keep]

        class_names = model.names
        for i in range(len(cls_ids)):
            raw_detections.append({
                'class_id': int(cls_ids[i]),
                'class_name': class_names[int(cls_ids[i])],
                'confidence': round(float(confs[i]), 4),
                'bbox': [round(float(v), 2) for v in boxes_xyxy[i]],
            })

    # ========== 收费站场景分析 ==========
    stats = {
        'person_count': 0,
        'vehicle_count': 0,
        'bicycle_count': 0,
        'motorcycle_count': 0,
        'bus_count': 0,
        'truck_count': 0,
    }
    warning_types = []

    for det in raw_detections:
        name = det['class_name']
        if name == 'person':
            stats['person_count'] += 1
        elif name in ['car', 'bus', 'truck']:
            stats['vehicle_count'] += 1
        elif name == 'bicycle':
            stats['bicycle_count'] += 1
        elif name == 'motorcycle':
            stats['motorcycle_count'] += 1

    # 拥堵判断
    if stats['vehicle_count'] > 5:
        warning_types.append('拥堵')

    # 行人闯入（示例区域）
    for det in raw_detections:
        if det['class_name'] == 'person':
            bbox = det['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            if 200 < cx < 600 and 100 < cy < 300:
                warning_types.append('行人闯入')
                break

    # 非机动车闯入
    for det in raw_detections:
        if det['class_name'] in ['bicycle', 'motorcycle']:
            warning_types.append('非机动车闯入')
            break

    # ========== 绘图 ==========
    annotated = frame.copy()
    color_map = {
        'person': (0, 255, 0),
        'car': (255, 0, 0),
        'bicycle': (0, 255, 255),
        'motorcycle': (255, 0, 255),
        'bus': (0, 100, 255),
        'truck': (0, 100, 200),
    }

    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = map(int, boxes_xyxy[i])
        cls_id = int(cls_ids[i])
        class_name = model.names[cls_id]
        conf = float(confs[i])
        color = color_map.get(class_name, (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{class_name} {conf:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    if warning_types:
        cv2.putText(annotated, f"⚠️ {' | '.join(warning_types)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    image_bytes = buffer.tobytes()

    logger.info(f"[图片检测] device={device_id} 检测到 {len(raw_detections)} 个目标, "
                f"预警: {warning_types if warning_types else '无'}")

    resp = HttpResponse(image_bytes, content_type='image/jpeg')
    resp['X-Detection-Data'] = json.dumps({
        'image_size': {'width': frame.shape[1], 'height': frame.shape[0]},
        'detections': raw_detections,
        'statistics': stats,
        'warning_types': warning_types,
    })
    return resp


@api_view(['POST'])
def create_task(request):
    """创建并启动视频流检测任务"""
    serializer = TaskCreateSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    stream_url = serializer.validated_data['stream_url']
    device_id = serializer.validated_data.get('device_id', '')
    fps = serializer.validated_data.get('fps')
    model_paths = serializer.validated_data.get('model_paths', ['yolov8s.pt'])

    task = Task(
        stream_url=stream_url,
        device_id=device_id,
        fps=fps,
        model_paths=model_paths,
    )
    tasks_store[task.task_id] = task
    logger.info(f"[创建任务] task_id={task.task_id}, url={stream_url}")

    start_engine(task)

    timeout = 15
    start = time.time()
    connected = False

    while time.time() - start < timeout:
        if task.status == 'running':
            connected = True
            break
        if task.status == 'failed':
            break
        time.sleep(0.5)

    if connected:
        return Response({
            'status': 'success',
            'task_id': task.task_id,
            'started': True,
            'message': '视频流连接成功，任务已启动',
        })

    stop_engine(task.task_id)
    task.status = 'failed'
    task.error = task.error or '连接超时'
    task.completed_at = datetime.now()

    return Response({
        'status': 'failed',
        'task_id': task.task_id,
        'started': False,
        'error': task.error,
        'message': '视频流连接失败，任务已删除',
    }, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET'])
def list_tasks(request):
    """查看所有任务"""
    tasks = [t.to_dict() for t in tasks_store.values()]
    return Response({
        'total': len(tasks),
        'tasks': tasks,
    })


@api_view(['GET'])
def get_task(request, task_id):
    """查看单个任务状态"""
    task_id = str(task_id)
    task = tasks_store.get(task_id)
    if not task:
        return Response({'detail': '任务不存在'}, status=status.HTTP_404_NOT_FOUND)
    return Response(task.to_dict())


@api_view(['POST'])
def stop_task(request, task_id):
    """停止并删除任务"""
    task_id = str(task_id)
    task = tasks_store.get(task_id)
    if not task:
        return Response({'detail': '任务不存在'}, status=status.HTTP_404_NOT_FOUND)

    stop_engine(task_id)
    task.status = 'stopped'
    task.completed_at = datetime.now()
    logger.info(f"[停止任务] task_id={task_id}")

    return Response({
        'status': 'success',
        'task_id': task_id,
        'message': '任务已停止并删除',
    })


@api_view(['POST'])
def upload_video(request):
    """
    上传视频文件到服务器
    """
    try:
        video_file = request.FILES.get('video')
        if not video_file:
            return Response({
                'success': False,
                'error': '请上传视频文件'
            }, status=status.HTTP_400_BAD_REQUEST)

        # 检查文件类型
        filename = video_file.name
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm']
        ext = os.path.splitext(filename)[1].lower()
        if ext not in allowed_extensions:
            return Response({
                'success': False,
                'error': f'不支持的视频格式: {ext}，请上传 {", ".join(allowed_extensions)} 格式'
            }, status=status.HTTP_400_BAD_REQUEST)

        # 保存视频文件到项目根目录
        file_path = os.path.join(settings.BASE_DIR, filename)
        
        # 如果文件已存在，重命名
        if os.path.exists(file_path):
            name, ext = os.path.splitext(filename)
            import time
            new_filename = f"{name}_{int(time.time())}{ext}"
            file_path = os.path.join(settings.BASE_DIR, new_filename)
            filename = new_filename

        with open(file_path, 'wb') as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        logger.info(f"[视频上传] 文件已保存: {filename}")

        return Response({
            'success': True,
            'filename': filename,
            'message': '视频上传成功'
        })

    except Exception as e:
        logger.exception(f"[视频上传] 失败: {e}")
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)