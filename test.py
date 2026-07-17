import requests
import json

# ========== 1. 测试健康检查 ==========
print("=" * 60)
print("1. 健康检查")
print("=" * 60)
try:
    resp = requests.get("http://127.0.0.1:9060/api/health/")
    print(f"状态码: {resp.status_code}")
    print(f"响应内容: {resp.json()}")
except Exception as e:
    print(f"错误: {e}")

# ========== 2. 测试图片检测 ==========
print("\n" + "=" * 60)
print("2. 图片检测")
print("=" * 60)

# 改成你自己的图片路径
image_path = r"D:\29477\Pictures\ims (2).webp"

try:
    url = "http://127.0.0.1:9060/api/detect/image/"
    files = {"image": open(image_path, "rb")}
    
    resp = requests.post(url, files=files)
    print(f"状态码: {resp.status_code}")
    
    # 从响应头读取检测数据
    detection_data = resp.headers.get("X-Detection-Data")
    if detection_data:
        data = json.loads(detection_data)
        print("\n📊 检测结果:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print("⚠️ 没有检测数据")
    
    # 保存标注图片
    with open("output.jpg", "wb") as f:
        f.write(resp.content)
    print("\n✅ 标注图片已保存为 output.jpg")
    
except FileNotFoundError:
    print(f"❌ 图片不存在: {image_path}")
except Exception as e:
    print(f"❌ 错误: {e}")



def test_create_task():
    """测试创建视频流任务"""
    print("=" * 60)
    print("3. 创建视频流任务")
    print("=" * 60)
    
    url = "http://127.0.0.1:9060/api/tasks/create/"
    data = {
        "stream_url": r"D:\yy\生成含汽车、行人和非机动车的视频.mp4",
        "fps": 2
    }
    resp = requests.post(url, json=data)
    print(f"状态码: {resp.status_code}")
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    # 先测健康检查和图片检测
    # 如果你有视频文件，取消下面一行的注释
    test_create_task()
    print("\n" + "=" * 60)
    print("✅ 测试完成！")