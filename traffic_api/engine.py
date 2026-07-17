import os
import json
import time
import queue
import shutil
import subprocess
import threading
import logging
from datetime import datetime

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from django.conf import settings

from . import utils
from .models import Task

# ============================================================
# 日志配置
# ============================================================
logger = logging.getLogger(__name__)

# 运行中的引擎池：task_id -> DetectionEngine 实例
CURRENT_ENGINES = {}

# 模型缓存：模型路径 -> YOLO 实例，避免重复加载
_model_cache = {}
_model_cache_refs = {}  # 模型路径 -> 引用计数
_model_cache_lock = threading.Lock()


class FFmpegReader:
    """基于 ffmpeg 的视频流读取器，替代 cv2.VideoCapture"""

    def __init__(self, url):
        self.url = url
        self.proc = None
        self._buf = b''
        self._opened = False
        self._stop = False
        self._chunk_queue = queue.Queue(maxsize=120)
        self._reader_thread = None
        self._stderr_thread = None

    def _pipe_reader(self):
        """后台线程：持续读取管道数据到队列"""
        try:
            while not self._stop:
                chunk = self.proc.stdout.read(65536)
                if not chunk:
                    break
                self._chunk_queue.put(chunk)
        except Exception:
            pass
        finally:
            self._chunk_queue.put(None)

    def _stderr_reader(self):
        """后台线程：读取 ffmpeg 错误日志"""
        try:
            for line in self.proc.stderr:
                line = line.decode('utf-8', errors='replace').strip()
                if line and not line.startswith('ffmpeg version'):
                    logger.info(f"[FFmpeg] {line}")
        except Exception:
            pass

    def isOpened(self):
        return self._opened

    def open(self):
        if not shutil.which('ffmpeg'):
            logger.warning(f"[FFmpeg] ffmpeg 未安装，跳过")
            return False

        cmd = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-loglevel', 'verbose',
            '-i', self.url,
            '-f', 'image2pipe',
            '-vcodec', 'mjpeg',
            '-qscale:v', '2',
            '-an',
            '-sn',
            '-',
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=10**8,
            )
            self._reader_thread = threading.Thread(target=self._pipe_reader, daemon=True)
            self._reader_thread.start()
            self._stderr_thread = threading.Thread(target=self._stderr_reader, daemon=True)
            self._stderr_thread.start()
            self._opened = True
            return True
        except Exception as e:
            logger.error(f"[FFmpeg] 启动失败: {e}")
            return False

    def read(self):
        if not self._opened or self.proc is None:
            return False, None

        idle_loops = 0
        while True:
            soi = self._buf.find(b'\xff\xd8')
            if soi >= 0:
                if soi > 0:
                    self._buf = self._buf[soi:]
                eoi = self._buf.find(b'\xff\xd9')
                if eoi >= 0:
                    jpeg_data = self._buf[:eoi + 2]
                    self._buf = self._buf[eoi + 2:]
                    frame = cv2.imdecode(
                        np.frombuffer(jpeg_data, dtype=np.uint8), cv2.IMREAD_COLOR
                    )
                    if frame is not None:
                        return True, frame
                    continue

            try:
                chunk = self._chunk_queue.get(timeout=1)
                idle_loops = 0
            except queue.Empty:
                idle_loops += 1
                if idle_loops == 30:
                    logger.warning(f"[FFmpeg] 等待首帧已 30 秒，ffmpeg 可能连接缓慢或出错")
                if self.proc and self.proc.poll() is not None:
                    logger.error(f"[FFmpeg] 进程已退出，返回码: {self.proc.returncode}")
                    return False, None
                continue
            if chunk is None:
                return False, None
            self._buf += chunk

    def release(self):
        self._stop = True
        if self.proc:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
        self._opened = False
        self._buf = b''


def get_model(model_path):
    """加载 YOLO 模型（带缓存 + 引用计数）"""
    if not model_path:
        raise ValueError("model_path 不能为空，请指定模型文件路径")
    with _model_cache_lock:
        if model_path not in _model_cache:
            logger.info(f"加载模型: {model_path}")
            from ultralytics import YOLO
            _model_cache[model_path] = YOLO(model_path)
            logger.info(f"模型加载完成: {model_path}")
        _model_cache_refs[model_path] = _model_cache_refs.get(model_path, 0) + 1
    return _model_cache[model_path]


def release_model(model_path):
    """释放模型引用，引用归零时从缓存移除"""
    with _model_cache_lock:
        refs = _model_cache_refs.get(model_path, 0)
        if refs <= 1:
            _model_cache_refs.pop(model_path, None)
            _model_cache.pop(model_path, None)
            logger.info(f"[模型] 已释放缓存: {model_path}")
        else:
            _model_cache_refs[model_path] = refs - 1


class DetectionEngine(threading.Thread):
    """
    YOLO 视频流检测引擎 - 收费站场景版
    每个任务对应一个独立的引擎线程
    """

    def __init__(self, task):
        super().__init__(daemon=True)
        self.task = task                      # 关联的任务对象
        self._stop_event = threading.Event()  # 停止信号
        self._cap = None                      # 视频流读取器
        # 收费站场景：车辆追踪器
        self._vehicle_trackers = {}            # 车辆追踪器（用于判断违停、逆行）
        self._next_tracker_id = 0              # 下一个跟踪器ID
        self._frame_counter = 0                # 帧计数器


    def stop(self):
        """请求引擎停止"""
        logger.info(f"引擎停止请求: task_id={self.task.task_id}")
        self._stop_event.set()
        if self._cap:
            self._cap.release()

    def is_stopped(self):
        """检查引擎是否已被要求停止"""
        return self._stop_event.is_set()

    @staticmethod
    def _box_iou(box1, box2):
        """计算两个框的 IoU（保留，用于追踪匹配）"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (area1 + area2 - inter + 1e-6)

    def _update_vehicle_trackers(self, vehicles, boxes_xyxy):
        """
        用 IoU 匹配更新帧间车辆跟踪器
        用于判断违停（停留超过 N 帧）和逆行（轨迹方向异常）
        """
        # vehicles: [(idx, class_name, confidence), ...]
        curr = [(boxes_xyxy[vi], class_name, conf) for vi, class_name, conf in vehicles]

        new_trackers = {}
        used_curr = set()

        # 匹配现有跟踪器与新帧的车辆
        for tid, t in self._vehicle_trackers.items():
            best_iou = 0.3
            best_ci = None
            for ci, (cbox, cname, cconf) in enumerate(curr):
                if ci in used_curr:
                    continue
                iou = self._box_iou(t['box'], cbox)
                if iou > best_iou:
                    best_iou = iou
                    best_ci = ci
            if best_ci is not None:
                cbox, cname, cconf = curr[best_ci]
                used_curr.add(best_ci)
                # 如果类别相同，计数器+1；否则重置为1
                same = (cname == t['class_name'])
                new_trackers[tid] = {
                    'box': cbox,
                    'class_name': cname,
                    'conf': cconf,
                    'count': t['count'] + 1 if same else 1,
                    'centers': t.get('centers', []) + [((cbox[0] + cbox[2]) / 2, (cbox[1] + cbox[3]) / 2)],
                    'pushed': t.get('pushed', False) and same,
                }
                # 只保留最近10个中心点
                if len(new_trackers[tid]['centers']) > 10:
                    new_trackers[tid]['centers'] = new_trackers[tid]['centers'][-10:]

        # 新增跟踪器
        for ci, (cbox, cname, cconf) in enumerate(curr):
            if ci not in used_curr:
                tid = self._next_tracker_id
                self._next_tracker_id += 1
                new_trackers[tid] = {
                    'box': cbox,
                    'class_name': cname,
                    'conf': cconf,
                    'count': 1,
                    'centers': [((cbox[0] + cbox[2]) / 2, (cbox[1] + cbox[3]) / 2)],
                    'pushed': False,
                }

        self._vehicle_trackers = new_trackers

    def _analyze_traffic(self, class_ids, class_names, confidences, boxes_xyxy, frame):
        """
        收费站场景交通分析
        检测：拥堵、行人闯入、非机动车闯入、违停、逆行
        """
        h, w = frame.shape[:2]

        # 获取类别ID映射
        names_lower = {v.lower(): k for k, v in class_names.items()}
        person_cid = names_lower.get('person')
        car_cid = names_lower.get('car')
        bicycle_cid = names_lower.get('bicycle')
        motorcycle_cid = names_lower.get('motorcycle')
        bus_cid = names_lower.get('bus')
        truck_cid = names_lower.get('truck')

        # ===== 1. 统计各类目标 =====
        statistics = {
            'person_count': 0,
            'vehicle_count': 0,
            'bicycle_count': 0,#自行车
            'tricycle_count': 0, #三轮车
            'motorcycle_count': 0,
            'bus_count': 0,
            'truck_count': 0,
            'stopped_vehicles': 0,     # 违停车辆数
            'wrong_way_vehicles': 0,   # 逆行车辆数
        }
        vehicles = []  # 用于追踪
        detections = []
        persons=[] #用于工作人员过滤

        ids = class_ids.tolist()
        for i, cid in enumerate(ids):
            class_name = class_names.get(cid, 'unknown')
            if class_name == 'person':
                statistics['person_count'] += 1
            elif class_name == 'car':
                statistics['vehicle_count'] += 1
                vehicles.append((i, class_name, float(confidences[i])))
            elif class_name == 'bicycle':
                statistics['bicycle_count'] += 1
                vehicles.append((i, class_name, float(confidences[i])))
            elif class_name == 'motorcycle':
                statistics['motorcycle_count'] += 1
                vehicles.append((i, class_name, float(confidences[i])))
            elif class_name == 'bus':
                statistics['bus_count'] += 1
                vehicles.append((i, class_name, float(confidences[i])))
            elif class_name == 'truck':
                statistics['truck_count'] += 1
                vehicles.append((i, class_name, float(confidences[i])))

            detections.append({
                'class_id': int(cid),
                'class_name': class_name,
                'confidence': round(float(confidences[i]), 4),
                'bbox': boxes_xyxy[i].tolist(),
            })

        # ===== 2. 更新车辆追踪器 =====
        self._update_vehicle_trackers(vehicles, boxes_xyxy)

        # ===== 3. 事件判断 =====
        warnings = []

        # 3.1 拥堵检测（车辆 > 阈值）
        if statistics['vehicle_count'] > 5:
            warnings.append({
                'type': 'congestion',
                'desc': f'拥堵，当前车辆数: {statistics["vehicle_count"]}',
                'confidence': 0.8
            })

        # 3.2 行人闯入
        # 工作人员判断：如果行人长时间在某个区域停留，可能是工作人员
        # 假设 y > 350 的区域是工作区，行人如果在这里停留超过30帧，可能是工作人员
        for det in detections:
            if det['class_name'] == 'person':
                bbox = det['bbox']
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                # 判断是否在禁止区域（车道区域）
                is_worker = False
                if cy > 350:
                    is_worker = True

                if is_worker:
                    continue
                if 200 < cx < 600 and 100 < cy < 300:
                    warnings.append({
                        'type': 'pedestrian_intrusion',
                        'desc': '行人闯入车道区域',
                        'confidence': det['confidence']
                    })
                    break

        # 3.3 非机动车闯入
        for det in detections:
            if det['class_name'] in ['bicycle', 'motorcycle']:
                warnings.append({
                    'type': 'non_motor_intrusion',
                    'desc': f'非机动车闯入: {det["class_name"]}',
                    'confidence': det['confidence']
                })
                break

        # 3.4 违停检测（车辆停留超过30帧，约1秒）
        for tid, tracker in self._vehicle_trackers.items():
            if tracker['count'] > 30 and not tracker.get('pushed', False):
                warnings.append({
                    'type': 'illegal_parking',
                    'desc': f'违停车辆 (ID:{tid}) 停留{tracker["count"]}帧',
                    'confidence': tracker['conf']
                })
                tracker['pushed'] = True

        # 3.5 逆行检测（基于轨迹方向，简化版：检查中心点是否向左移动）
        for tid, tracker in self._vehicle_trackers.items():
            centers = tracker.get('centers', [])
            if len(centers) >= 5:
                # 检查最近5个点的水平移动方向
                dx_sum = 0
                for i in range(len(centers) - 1):
                    dx_sum += centers[i+1][0] - centers[i][0]
                if dx_sum < -20:  # 持续向左移动
                    warnings.append({
                        'type': 'wrong_way',
                        'desc': f'逆行车辆 (ID:{tid})',
                        'confidence': tracker['conf']
                    })

        # ===== 4. 更新任务状态 =====
        self.task.warnings = warnings
        self.task.statistics = statistics
        self.task.detections = detections

        return detections, warnings, statistics

    def run(self):
        """
        引擎主循环：逐帧读取视频流 -> YOLO 检测 -> 异常判断 -> 预警推送
        """
        task_id = self.task.task_id
        logger.info(f"[引擎] 任务 {task_id} 引擎线程启动")

        try:
            # 标记任务为运行中
            self.task.status = 'running'
            self.task.started_at = datetime.now()
            logger.info(f"[引擎] 任务 {task_id} 状态已更新为 running")

            # 加载所有 YOLO 模型
            if not self.task.model_paths:
                raise ValueError("至少需要一个模型文件路径")
            models = []
            all_class_names = {}
            for mp in self.task.model_paths:
                m = get_model(mp)
                models.append(m)
                all_class_names.update(m.names)

            class_names = all_class_names
            logger.info(f"[引擎] 模型个数: {len(models)}, 类别数: {len(class_names)}")

            # 使用 OpenCV VideoCapture 读取视频流
            self._cap = cv2.VideoCapture(self.task.stream_url)
            if not self._cap.isOpened():
                logger.warning(f"[引擎] OpenCV 打开失败，尝试 FFmpegReader: {self.task.stream_url}")
                reader = FFmpegReader(self.task.stream_url)
                if reader.open():
                    self._cap = reader
                else:
                    raise RuntimeError(f"无法打开视频流: {self.task.stream_url}")
                    

            logger.info(f"[引擎] 视频流连接成功: {self.task.stream_url}")

            # 抽帧频率控制
            fps = self.task.fps
            frame_interval = 1.0 / fps if fps else 0
            last_time = time.time()

            frame_count = 0
            reconnect_delay = 1
            while not self.is_stopped():
                ret, frame = self._cap.read()
                if not ret:
                    logger.warning(f"[引擎] 读取超时，尝试重连... (等待 {reconnect_delay}s)")
                    self._cap.release()
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 30)
                    self._cap = cv2.VideoCapture(self.task.stream_url)
                    if self._cap.isOpened():
                        logger.info(f"[引擎] 重连成功")
                        reconnect_delay = 1
                        continue
                    else:
                        logger.warning(f"[引擎] 重连失败，继续等待...")
                        continue

                reconnect_delay = 1
                self.task.total_frames += 1

                # 如果设置了抽帧频率，跳过中间帧
                if fps:
                    now = time.time()
                    if now - last_time < frame_interval:
                        continue
                    last_time = now

                # 多模型推理并聚合检测结果
                all_boxes = []
                all_cls = []
                all_conf = []
                for model in models:
                    results = model(frame, conf=0.25, iou=0.7, verbose=False)
                    detections = results[0].boxes
                    if detections is not None and len(detections) > 0:
                        boxes = detections.xyxy.cpu().numpy() if hasattr(detections.xyxy, 'cpu') else detections.xyxy
                        cls = detections.cls.cpu().numpy() if hasattr(detections.cls, 'cpu') else detections.cls
                        conf = detections.conf.cpu().numpy() if hasattr(detections.conf, 'cpu') else detections.conf
                        all_boxes.append(boxes)
                        all_cls.append(cls)
                        all_conf.append(conf)

                frame_count += 1
                if all_boxes:
                    self.task.result_frames_count += 1
                    boxes_xyxy = np.vstack(all_boxes)
                    cls_ids = np.concatenate(all_cls)
                    confs = np.concatenate(all_conf)

                    # 同类 NMS去重
                    from . import utils
                    keep = utils.class_specific_nms(boxes_xyxy, confs, cls_ids, iou_thresh=0.6)
                    if len(keep) < len(boxes_xyxy):
                        boxes_xyxy = boxes_xyxy[keep]
                        cls_ids = cls_ids[keep]
                        confs = confs[keep]
                    self.task.detections_count += len(cls_ids)
                    
                    detections, warnings, statistics = self._analyze_traffic(cls_ids, class_names, confs, boxes_xyxy, frame)
                    self.task.warnings = warnings
                    self.task.statistics = statistics
                    self.task.detections = detections


                self.task.processed_frames += 1
                if frame_count % 200 == 0:
                    logger.info(f"[引擎] 任务 {task_id} 进度: "
                                f"总帧数={self.task.total_frames} "
                                f"已处理={self.task.processed_frames} "
                                f"检测数={self.task.detections_count}")

            # 释放视频流
            self._cap.release()
            logger.info(f"[引擎] 任务 {task_id} 视频流已释放")

            # 正常结束：标记为 completed
            if not self.is_stopped():
                self.task.status = 'completed'
                self.task.completed_at = datetime.now()
                logger.info(f"[引擎] 任务 {task_id} 已完成")

        except Exception as e:
            logger.exception(f"[引擎] 任务 {task_id} 运行异常: {e}")
            self.task.status = 'failed'
            self.task.error = str(e)
            self.task.completed_at = datetime.now()

        finally:
            # 清理
            CURRENT_ENGINES.pop(task_id, None)
            for mp in self.task.model_paths:
                release_model(mp)
            logger.info(f"[引擎] 任务 {task_id} 引擎线程结束")


    def start_engine(task):
        """启动检测引擎"""
        engine = DetectionEngine(task)
        CURRENT_ENGINES[task.task_id] = engine
        engine.start()
        logger.info(f"[引擎] 任务 {task.task_id} 引擎已启动")
        return engine


    def stop_engine(task_id):
        """停止指定任务的检测引擎"""
        engine = CURRENT_ENGINES.pop(task_id, None)
        if engine:
            engine.stop()
            logger.info(f"[引擎] 任务 {task_id} 引擎已停止")
            return True
        logger.warning(f"[引擎] 任务 {task_id} 引擎不存在，无需停止")
        return False

# ============================================================
# 引擎管理函数
# ============================================================

def start_engine(task):
    """启动检测引擎"""
    engine = DetectionEngine(task)
    engine.start()
    logger.info(f"[引擎] 任务 {task.task_id} 引擎已启动")
    return engine


def stop_engine(task_id):
    """停止指定任务的检测引擎"""
    # 简化版：从 CURRENT_ENGINES 中移除并停止
    engine = CURRENT_ENGINES.pop(task_id, None)
    if engine:
        engine.stop()
        logger.info(f"[引擎] 任务 {task_id} 引擎已停止")
        return True
    logger.warning(f"[引擎] 任务 {task_id} 引擎不存在")
    return False