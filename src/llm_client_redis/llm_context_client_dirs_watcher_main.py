import argparse
import logging
import os
import random

from llm_client_redis.llm_client import LLMClientRedis
from langchain_core.messages.human import HumanMessage
from langchain_core.messages.ai import AIMessage
from llm_client_redis.tools.output_tools import OutputTools
from time import sleep
# from llm_tokenizers.deepseek_tokenizer import DeepSeekTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class ContextDirsWatcher:
    """
    同时监控多目录监视程序,每一个执行周期均对监控目录进行重新洗牌重新决定其执行顺序; 在每一次执行中, 均会对每一个监控目录中的文件指定后缀名的文件进行查找，如果找到相应的文件，则加载文件的内容，并从当前文件夹的名称
    中获取到对应的模型名称，然后将文本发送到相应的大模型，再将输出结果放到本文件夹的 results 目录中，文件名与输入文件名相同，后缀名改为 _result.txt
    """

    focusDirs: list[str]
    outpuSubDirName: str
    fileSuffix: list[str]
    contextFileSuffix: list[str]
    suffle: bool
    sleepInterval: int

    def __init__(self, focus_dirs: list[str], 
                 output_sub_dir_name: str, 
                 file_suffix: list[str] = [".md", ".txt", "pro", ".prompt"], 
                 context_file_suffix: list[str] = [".context"], 
                 suffle: bool = True, 
                 sleep_interval: int = 3):
        """
        初始化
        :param focus_dirs: 监控的目录列表
        :param output_sub_dir_name: 输出子目录名称
        :param file_suffix: 文件后缀名
        :param context_file_suffix: 上下文文件后缀名
        :param suffle: 是否洗牌
        :param sleep_interval: 执行间隔
        """
        self.focusDirs = focus_dirs
        self.outpuSubDirName = output_sub_dir_name
        self.fileSuffix = file_suffix
        self.contextFileSuffix = context_file_suffix
        self.suffle = suffle
        self.sleepInterval = sleep_interval

        logging.info(f"初始化监控目录: {self.focusDirs}")
        logging.info(f"初始化输出子目录名称: {self.outpuSubDirName}")
        logging.info(f"初始化文件后缀名: {self.fileSuffix}")
        logging.info(f"初始化上下文文件后缀名: {context_file_suffix}")
        logging.info(f"初始化随机文件夹: {self.suffle}")
        logging.info(f"初始化执行间隔(s): {self.sleepInterval}")
        pass

    def watchAndDoInDirs(self, overwrite: bool = False) -> None:
        """
        监控目录中的文件，并执行相应的操作
        :param overwrite: 是否覆盖输出文件, 默认为 False
        """
        # sourceDirs 包含 self.focusDirs 中的所有目录，但不是同一个列表，需要另建列表
        sourceDirs: list[str] = self.focusDirs.copy()

        doneWork:bool = False

        while True:
            if self.suffle:
                random.shuffle(sourceDirs)
            
            for sourceDir in sourceDirs:

                # model 从 sourceDir 中获取，找出文件夹名称最后的 _，移除其为所有内容，则为模型名称
                model: str = os.path.basename(sourceDir)

                model = model[0: model.rfind("_context")]

                logging.info(f"Processing dir: {sourceDir}, model: {model}")

                result: str = self.llmSendFilesInDir(sourceDir, self.outpuSubDirName, model, self.fileSuffix, overwrite)

                if result is not None and result != "":
                    doneWork = True
                    logging.info(f"Processing dir: {sourceDir}, model: {model}, result: {result}, sleep 3 seconds.")
                    sleep(3)

                else:
                    logging.info(f"There is no file to process in this dir {sourceDir}")

            if doneWork:
                doneWork = False
            else:
                logging.info(f"Sleeping for {self.sleepInterval} seconds")
                sleep(self.sleepInterval)
        pass

    def llmSendFilesInDir(self, 
                          source_dir: str, 
                          output_sub_dir_name: str, 
                          model: str, 
                          file_suffix: list[str], 
                          overwrite: bool = False,
                          max_retry: int = 6,
                          retry_sleep_interval: int = 20) -> str:
        """
        在指定目录中查找指定后缀名的文件，并使用指定的模型发送文件内容到大模型，并将结果保存到指定目录中.
        其中，可以从 source_dir 中获取模型名称，因为它均是 model_数字 的形式。
        output_sub_dir_name 是输出目录的子目录名称

        :param source_dir: 源目录
        :param output_sub_dir_name: 输出子目录名称
        :param model: 模型名称
        :param file_suffix: 文件后缀名
        :param overwrite: 是否覆盖输出文件, 默认为 False
        :param max_retry: 最大重试次数, 默认为 6
        :param retry_sleep_interval: 重试间隔时间, 默认为 20 秒
        """

        retry_count: int = 0

        while retry_count < max_retry:
            try:
                files: list = os.listdir(source_dir)
                break
            except FileNotFoundError as e:
                logging.warning(f"Processing dir: {source_dir}, model: {model}, but dir not found, may be the network disk faild, wait {retry_sleep_interval} seconds and retry. error: {e}")
                sleep(retry_sleep_interval)
                retry_count += 1

        if files is None or len(files) == 0:
            return ""

        random.shuffle(files)                    

        # 需要进行处理的文本
        llm: LLMClientRedis = LLMClientRedis()

        for file in files:

            for fileSubffix in file_suffix:

                if str.lower(file).endswith(f"{fileSubffix}"):

                    fileBaseName: str = file[0: file.rfind(fileSubffix)]

                    _contextFileSuffix: str = None

                    # 上下文文件
                    contextFile: str = None

                    # 如果检测不到同名的 .context 文件, 则报错
                    for contextFileSuffix in self.contextFileSuffix:

                        contextFile = os.path.join(source_dir, fileBaseName + contextFileSuffix)

                        if os.path.exists(contextFile) and os.path.isfile(contextFile):
                            _contextFileSuffix = contextFileSuffix
                            break
                        else:
                            logging.error(f"Processing file: {file}, but not found context file, raise error.")
                            raise FileNotFoundError(f"Processing file: {file}, but not found context file, raise error.")
                    
                    # 将原文件名重命名加上 .working 后缀表示正在执行
                    try:
                        os.rename(os.path.join(source_dir, file), os.path.join(source_dir, file + ".working"))
                        logging.info(f"Processing file: {file}, and rename to {file + '.working'}")
                    except FileNotFoundError as e:
                        logging.warning(f"Processing file: {file}, and rename to {file + '.working'}, "
                                        f"but file not found, may be already processed, continue. error: {e}")
                        continue
                    except PermissionError as e:
                        logging.warning(f"Processing file: {file}, and rename to {file + '.working'}, "
                                        f"but file not found, may be already processed, continue. error: {e}")
                        continue

                    workingFile: str = file + ".working"

                    # 加载提示词的内容
                    _text: str = None

                    try:
                        with open(os.path.join(source_dir, workingFile), "r", encoding="utf-8") as f:
                            _text: str = f.read()
                    except PermissionError as e:
                        logging.warning(f"Processing file: {file}, and read file, but file not found, may be already processed, continue. error: {e}")
                        continue
                    
                    # 加载上下文内容
                    _context: str = None

                    try:
                        with open(os.path.join(source_dir, contextFile), "r", encoding="utf-8") as f:
                            _context: str = f.read()
                    except PermissionError as e:
                        logging.warning(f"Processing context file: {file}, and read file, but file not found, may be already processed, continue. error: {e}")
                        continue

                    # token_count: int = DeepSeekTokenizer.tokens_len(_text)

                    # logging.info(f"Processing file: {file}, token count: {token_count}")

                    msg: HumanMessage = HumanMessage(content=_text)
                    assistMsg: AIMessage = AIMessage(content=_context)
                    
                    _result: str = ""

                    # model="deepseek_r1" 或者 model="huawei_deepseek_r1_32k" 或者 model="huawei_DeepSeek-R1-32K-0528"
                    for _chunk in llm.request_stream(messages=[msg, assistMsg], 
                                                     model=model,
                                                     ontinue_final_message=True, 
                                                     add_generation_prompt=False                                                     
                                                     ):
                        _result += _chunk
                        print(_chunk, end="", flush=True)
                    
                    _only_json = OutputTools.only_json(_context + _result)

                    # 检查  是否存在，不存在则创建
                    if not os.path.exists(os.path.join(source_dir, output_sub_dir_name)):

                        while retry_count < max_retry:
                            try:
                                os.makedirs(os.path.join(source_dir, output_sub_dir_name))
                                logging.info(f"Create output sub dir: {output_sub_dir_name}")
                            # 由于使用网络磁盘的网络故障导致,此处应进行20秒的等待,然后再进行重试
                            except FileNotFoundError as e:
                                logging.warning(f"{retry_count}/{max_retry} Create output sub dir: {output_sub_dir_name}, "
                                                f"but file not found, may be network error, sleep {retry_sleep_interval} seconds, and retry. error: {e}")
                                sleep(retry_sleep_interval)
                                retry_count += 1

                    with open(os.path.join(source_dir, output_sub_dir_name, file + ".json"), "w", encoding="utf-8") as f:
                        f.write(_only_json)
                    logging.info(f"Processed json output file: {file}")

                    try:
                        # 然后将 .working 文件移动到 output_sub_dir_name 目录中，并移除 .working 后缀
                        os.rename(os.path.join(source_dir, workingFile), os.path.join(source_dir, output_sub_dir_name, file))
                        logging.info(f"Processed file: {file}, and move to {output_sub_dir_name} directory")
                    except FileNotFoundError as e:
                        logging.warning(f"Processed file: {file}, and move to {output_sub_dir_name} directory, but file not found, error: {e}, may be the file has been deleted by other process, and skip.")
                    
                    return os.path.join(source_dir, output_sub_dir_name, file)

            logging.info(f"File: {file} is not in {file_suffix}")
        
        return ""


def main():
    """
    恒常的入口
    """
    # 设置参数解析器
    parser = argparse.ArgumentParser(description="定期监测文录提示词(使用上下文引导生成结果),前文同时会保存到结果中")

    # 提示词生成路径
    parser.add_argument('-p', '--prompt_paths', type=str, required=False, help=f'指定监控的目录,多个目录使用英文逗号分隔,与-w选项不可同存,注意目录的名称为需要使用的模型名称加上_数字; 例如: /path/to/model_1,/path/to/model_2,...')
    parser.add_argument('-o', '--output_path', type=str, default="results", help=f'指定输出子目录名称，默认: results')
    parser.add_argument('-i', '--interval', type=int, default=60, help=f'指定监控的间隔时间，默认: 60 秒')
    parser.add_argument('-r', '--random-start', action='store_true', help=f'是否随机开始，默认: False')

    parser.add_argument('-w', '--watch-dir', type=str, required=False, help=f'指定监控目录，此目录下的所有子目录将会被纳入监控，与 -p 选项不可共存')

    args = parser.parse_args()

    prompt_path: str = args.prompt_paths  # 获取提示词生成路径
    output_path: str = args.output_path  # 获取输出路径
    interval: int = args.interval  # 获取间隔时间
    watch_dir: str = args.watch_dir  # 获取监控目录
    random_start: bool = args.random_start  # 获取随机开始

    # 打印参数用于调试
    logging.info(f"Prompt Path: {prompt_path}")
    logging.info(f"Output Path: {output_path}")
    logging.info(f"Interval: {interval}")
    logging.info(f"Watch Dir: {watch_dir}")
    logging.info(f"Random Start: {random_start}")

    if random_start:
        random_time: int = random.randint(0, interval)
        logging.info(f"随机开始, 随机时间为: {random_time} 秒")
        sleep(random_time)

    if prompt_path is None and watch_dir is None:
        logging.error("请指定监控的目录 -p 或者监控的目录 -w")
        exit(1)

    if prompt_path is not None and watch_dir is not None:
        logging.error("请指定监控的目录 -p 或者监控的目录 -w, 不能同时指定")
        exit(1)
    
    focusDirs: list[str] = None

    if prompt_path is not None:                             
        focusDirs = prompt_path.split(",")
    
    if watch_dir is not None:
        # 此处需要每个 watch_dir 的子目录，均需要与  watch_dir join 在一起
        focusDirs = [os.path.join(watch_dir, d) for d in os.listdir(watch_dir) if os.path.isdir(os.path.join(watch_dir, d))]

    contextDirWatcher: ContextDirsWatcher = ContextDirsWatcher(focusDirs, output_path, sleep_interval=interval)

    contextDirWatcher.watchAndDoInDirs(overwrite=False)

    logging.info("处理完成")


if __name__ == '__main__':
    
    main()
