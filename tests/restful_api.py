# 使用以下的命令开启 fast api
# cd src
# uvicorn llm_client_redis.llm_restful_client_main:app --host 127.0.0.1 --port 8080 --reload

import requests
import json


def demo_stream_request():
    """
    演示如何调用 /stream 接口
    """

    # FastAPI 服务的基础 URL
    base_url = "http://127.0.0.1:8080"  # 根据实际部署地址修改

    # 示例0: 基本请求
    print("=== 示例0: 基本请求 ===")
    response = requests.get(f"{base_url}/models")

    if response.status_code == 200:
        print("响应内容:")
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                print(chunk.decode('utf-8'), end='', flush=True)
        print("\n")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")

    # 示例1: 基本请求
    print("=== 示例1: 基本请求 ===")
    payload = {
        "message": "你好，世界！",
        "model": "home_qwen3:32b"
    }

    response = requests.get(f"{base_url}/stream", json=payload, stream=True)

    if response.status_code == 200:
        print("响应内容:")
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                print(chunk.decode('utf-8'), end='', flush=True)
        print("\n")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")

    # 示例2: 包含 system_message 的请求
    print("=== 示例2: 包含 system_message 的请求 ===")
    payload_with_system = {
        "message": "请写一首关于春天的诗",
        "model": "home_qwen3:32b",
        "system_message": "你是一个富有诗意的助手，擅长写诗"
    }

    response = requests.post(f"{base_url}/stream", json=payload_with_system, stream=True)

    if response.status_code == 200:
        print("响应内容:")
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                print(chunk.decode('utf-8'), end='', flush=True)
        print("\n")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")

    # 示例3: 使用默认模型（不指定 model 参数）
    print("=== 示例3: 使用默认模型 ===")
    payload_default_model = {
        "message": "什么是人工智能？"
    }

    response = requests.post(f"{base_url}/stream", json=payload_default_model, stream=True)

    if response.status_code == 200:
        print("响应内容:")
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                print(chunk.decode('utf-8'), end='', flush=True)
        print("\n")
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(f"错误信息: {response.text}")


def demo_stream_request_with_error_handling():
    """
    演示带错误处理的请求
    """
    base_url = "http://127.0.0.1:8080"

    # 测试缺少必需参数的情况
    print("=== 示例4: 错误处理示例 ===")
    invalid_payload = {
        "model": "home_qwen3:32b"
        # 故意缺少必需的 'message' 字段
    }

    response = requests.post(f"{base_url}/stream", json=invalid_payload)

    if response.status_code == 422:
        print("参数验证失败:")
        print(f"状态码: {response.status_code}")
        print(f"错误详情: {response.json()}")
    else:
        print(f"意外的响应状态码: {response.status_code}")


if __name__ == "__main__":
    # 运行演示
    try:
        demo_stream_request()
        demo_stream_request_with_error_handling()
    except requests.exceptions.ConnectionError:
        print("无法连接到服务器，请确保 FastAPI 服务正在运行")
    except Exception as e:
        print(f"发生错误: {e}")
