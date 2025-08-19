import logging

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator
from .llm_client import LLMClientRedis
from langchain_core.messages import SystemMessage, HumanMessage
from typing import List, Optional

app = FastAPI()

# 全局LLM客户端实例
llm: LLMClientRedis = LLMClientRedis()


class StreamRequest(BaseModel):
    message: str
    model: Optional[str] = None
    system_message: Optional[str] = None

    @field_validator('system_message')
    def validate_system_message(cls, v):
        if v == "":
            return None
        return v

class StreamMessagesRequest(BaseModel):
    messages: List[dict]  # 包含role和content的字典列表
    model: Optional[str] = None

@app.post("/stream")
async def stream(request: StreamRequest):
    """
    处理简单的文本流式请求
    """
    try:
        # 构建消息列表
        messages = []
        if request.system_message:
            messages.append(SystemMessage(request.system_message))
        messages.append(HumanMessage(request.message))

        # 使用默认模型如果未指定
        model = request.model or "home_qwen3:32b"
        logging.info(f"model: {model}")

        # 创建异步生成器包装器
        async def generate():
            try:
                # 调用request_stream方法
                for chunk in llm.request_stream(
                        messages=messages,
                        model=model
                ):
                    yield chunk
            except Exception as e:
                yield f"Error: {str(e)}"

        return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream/messages")
async def stream_messages(request: StreamMessagesRequest):
    """
    处理完整消息列表的流式请求
    """
    try:
        # 转换消息格式
        messages = []
        for msg in request.messages:
            if msg["role"] == "system":
                messages.append(SystemMessage(msg["content"]))
            elif msg["role"] == "human":
                messages.append(HumanMessage(msg["content"]))
            # 可以添加更多角色类型

        # 使用默认模型如果未指定
        model = request.model or "home_qwen3:32b"

        # 创建异步生成器包装器
        async def generate():
            try:
                # 调用request_stream方法
                for chunk in llm.request_stream(
                        messages=messages,
                        model=model
                ):
                    yield chunk
            except Exception as e:
                yield f"Error: {str(e)}"

        return StreamingResponse(generate(), media_type="text/plain")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models.json")
@app.post("/models.json")
async def post_models():
    """
    获取可用模型列表
    """
    try:
        # 这里需要访问LLMResourcesTools来获取模型列表
        # 由于是私有属性，我们可以通过其他方式获取
        return llm.llm_resources_tools.list_llm_def()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """
    获取可用模型列表
    """
    try:
        # 这里需要访问LLMResourcesTools来获取模型列表
        # 由于是私有属性，我们可以通过其他方式获取
        return llm.llm_resources_tools.list_llm_def()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/demo.json")
async def post_demo():
    """
    获取可用模型列表
    """
    try:
        return ["Hello World!"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))