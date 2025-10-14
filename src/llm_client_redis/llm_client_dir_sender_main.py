import argparse
import logging
import os

from llm_client_redis.llm_client import LLMClientRedis
from langchain_core.messages.human import HumanMessage
from llm_client_redis.tools.output_tools import OutputTools
# from llm_tokenizers.deepseek_tokenizer import DeepSeekTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def llmSendFilesInDir(source_dir: str, output_dir: str, model: str, file_suffix: str, overwrite: bool = False):

    files: list = os.listdir(source_dir)

    # 需要进行处理的文本
    llm: LLMClientRedis = LLMClientRedis()

    for file in files:
        if file.endswith(f".{file_suffix}"):
            _text: str = None
            with open(os.path.join(source_dir, file), "r", encoding="utf-8") as f:
                _text: str = f.read()

            # token_count: int = DeepSeekTokenizer.tokens_len(_text)

            # logging.info(f"Processing file: {file}, token count: {token_count}")

            msg: HumanMessage = HumanMessage(content=_text)
            
            _result: str = ""

            # model="deepseek_r1" 或者 model="huawei_deepseek_r1_32k" 或者 model="huawei_DeepSeek-R1-32K-0528"
            for _chunk in llm.request_stream(messages=[msg], model=model):
                _result += _chunk
                print(_chunk, end="", flush=True)
            
            _only_json = OutputTools.only_json(_result)

            with open(os.path.join(output_dir, file + ".json"), "w", encoding="utf-8") as f:
                f.write(_only_json)
            logging.info(f"Processed json output file: {file}")

def main():
    """
    恒常的入口
    """
    # 设置参数解析器
    parser = argparse.ArgumentParser(description="处理诊断和操作讨论的流程控制")
    parser.add_argument('-m', '--model', type=str, default='deepseek_r1',
                        help='指定使用的模型名称，默认为 deepseek_r1')

    # 提示词生成路径
    parser.add_argument('-p', '--prompt_path', type=str, required=True, help=f'指定提示词来源路径')
    parser.add_argument('-o', '--output_path', type=str, required=True, help=f'指定输出路径')
    parser.add_argument('-s', '--file-suffix', type=str, default='txt', help=f'指定文件后缀名，默认为 .txt')

    args = parser.parse_args()

    prompt_path: str = args.prompt_path  # 获取提示词生成路径
    output_path: str = args.output_path  # 获取输出路径
    model_name: str = args.model  # 获取模型名称
    file_suffix: str = args.file_suffix  # 获取文件后缀名

    # 打印参数用于调试
    logging.info(f"Model: {model_name}")
    logging.info(f"Prompt Path: {prompt_path}")
    logging.info(f"Output Path: {output_path}")

    llmSendFilesInDir(prompt_path, output_path, model=model_name, file_suffix=file_suffix)

    logging.info("处理完成")


if __name__ == '__main__':
    
    main()
