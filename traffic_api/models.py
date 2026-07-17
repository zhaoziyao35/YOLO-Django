import uuid
from datetime import datetime


class Task:
    def __init__(self, stream_url, device_id='', fps=None, model_paths=None):
        self.task_id = str(uuid.uuid4())
        self.stream_url = stream_url
        self.device_id = device_id
        self.fps = fps
        self.model_paths = model_paths or ['yolov8s.pt']
        self.status = 'pending'  # pending, running, completed, failed, stopped
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.total_frames = 0
        self.processed_frames = 0
        self.result_frames_count = 0
        self.detections_count = 0
        self.error = None
        # 收费站场景字段
        self.warnings = []
        self.statistics = {}
        self.detections = []

    def to_dict(self, fields=None):
        data = {
            'task_id': self.task_id,
            'status': self.status,
            'stream_url': self.stream_url,
            'device_id': self.device_id,
            'fps': self.fps,
            'model_paths': self.model_paths,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'total_frames': self.total_frames,
            'processed_frames': self.processed_frames,
            'result_frames_count': self.result_frames_count,
            'detections_count': self.detections_count,
            'warnings': self.warnings,
            'statistics': self.statistics,
            'error': self.error,
        }
        if fields:
            return {k: v for k, v in data.items() if k in fields}
        return data


# 内存存储
tasks_store = {}