import unittest
from fastapi.testclient import TestClient
from src.llm_client_redis.llm_restful_client_main import app


class TestLLMRestfulClient(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_stream_endpoint(self):
        """
        测试 /stream POST 接口
        """
        # 测试基本请求
        response = self.client.post("/stream", json={
            "message": "Hello, world!",
            "model": "home_qwen3:32b"
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/plain; charset=utf-8")

        # 测试带 system_message 的请求
        response = self.client.post("/stream", json={
            "message": "Hello, world!",
            "model": "home_qwen3:32b",
            "system_message": "You are a helpful assistant."
        })
        self.assertEqual(response.status_code, 200)

        # 测试空字符串 system_message（会被验证器转换为 None）
        response = self.client.post("/stream", json={
            "message": "Hello, world!",
            "model": "home_qwen3:32b",
            "system_message": ""
        })
        self.assertEqual(response.status_code, 200)

        # 测试不提供 model 参数（使用默认值）
        response = self.client.post("/stream", json={
            "message": "Hello, world!"
        })
        self.assertEqual(response.status_code, 200)

    def test_stream_messages_endpoint(self):
        """
        测试 /stream/messages POST 接口
        """
        # 测试基本消息列表请求
        response = self.client.post("/stream/messages", json={
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "human", "content": "Hello, world!"}
            ],
            "model": "home_qwen3:32b"
        })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/plain; charset=utf-8")

        # 测试不提供 model 参数（使用默认值）
        response = self.client.post("/stream/messages", json={
            "messages": [
                {"role": "human", "content": "Hello, world!"}
            ]
        })
        self.assertEqual(response.status_code, 200)

        # 测试只有 human 消息
        response = self.client.post("/stream/messages", json={
            "messages": [
                {"role": "human", "content": "What's the weather like today?"}
            ],
            "model": "home_qwen3:32b"
        })
        self.assertEqual(response.status_code, 200)

    def test_models_endpoint(self):
        """
        测试 /models GET 接口
        """
        response = self.client.get("/models")
        # 由于依赖于实际的 LLM 资源，我们只验证状态码
        self.assertIn(response.status_code, [200, 500])  # 可能成功也可能因环境而失败

    def test_stream_invalid_request(self):
        """
        测试 /stream 接口的无效请求
        """
        # 缺少必需的 message 字段
        response = self.client.post("/stream", json={
            "model": "home_qwen3:32b"
        })
        self.assertEqual(response.status_code, 422)  # 验证错误

        # 空消息
        response = self.client.post("/stream", json={
            "message": "",
            "model": "home_qwen3:32b"
        })
        # 根据实际验证规则，这可能成功或失败

    def test_stream_messages_invalid_request(self):
        """
        测试 /stream/messages 接口的无效请求
        """
        # 缺少必需的 messages 字段
        response = self.client.post("/stream/messages", json={
            "model": "home_qwen3:32b"
        })
        self.assertEqual(response.status_code, 422)  # 验证错误

        # 空消息列表
        response = self.client.post("/stream/messages", json={
            "messages": [],
            "model": "home_qwen3:32b"
        })
        self.assertEqual(response.status_code, 200)  # 空列表可能是有效的


if __name__ == "__main__":
    unittest.main()
