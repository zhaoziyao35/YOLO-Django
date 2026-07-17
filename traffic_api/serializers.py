from rest_framework import serializers


class TaskCreateSerializer(serializers.Serializer):
    device_id = serializers.CharField(required=False, default='', allow_blank=True)
    stream_url = serializers.CharField(required=True)
    fps = serializers.IntegerField(required=False, allow_null=True, default=None)
    model_path = serializers.CharField(required=False, default='', allow_blank=True)
    model_paths = serializers.ListField(
        child=serializers.CharField(allow_blank=True),
        required=False,
        default=list,
    )


class ImageDetectSerializer(serializers.Serializer):
    """图片检测请求参数校验"""
    image = serializers.ImageField(required=True)                        # 上传的图片文件
    model_path = serializers.CharField(required=False, default='', allow_blank=True)  # 模型路径，空则使用默认模型
    device_id = serializers.CharField(required=False, default='image_detect', allow_blank=True)  # 设备标识
    conf_threshold = serializers.FloatField(required=False, default=0.25)  # 检测置信度阈值
