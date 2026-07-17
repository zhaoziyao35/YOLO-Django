from django.urls import path
from . import views  

urlpatterns = [
    path('detect/image/', views.detect_image, name='detect-image'),
    path('health/', views.health_check, name='health'),

        # 视频流任务管理
    path('upload/video/', views.upload_video, name='upload-video'),
    path('tasks/create/', views.create_task, name='create-task'),
    path('tasks/', views.list_tasks, name='list-tasks'),
    path('tasks/<str:task_id>/', views.get_task, name='get-task'),
    path('tasks/<str:task_id>/stop/', views.stop_task, name='stop-task'),
]