call conda activate llm_client_redis

uvicorn src.llm_client_redis.llm_restful_client_main:app --reload