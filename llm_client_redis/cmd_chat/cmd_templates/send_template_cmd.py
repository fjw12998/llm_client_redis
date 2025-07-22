import argparse
import json
import os
import re
from typing import List

from langchain_core.messages import HumanMessage

from ..command_def import Command, AbstractChatSession


class SendTemplateCommand(Command):
    """
    实现模板 + 数据 JSON 的组合发送功能
    """

    def __init__(self):
        super().__init__("send", [], "参数-f <file_path> 发送模版文件 或 "
                                     "-t <text_block_seq>从 text_block 中获取相应序号的文本进行发送，"
                                     "第二个参数 --json <json_file_path or json_file_dir>，则从第二个参数指向的文件或文件夹中，"
                                     "加载 JSON 数据用于合成模板，再发送llm")

    def match(self, input_str: str, session: AbstractChatSession) -> bool:
        """ 判断是否匹配 """

        # 创建一个正则表达式判断输入
        pattern: str = rf"{self.name}\s+(-f|-t)\s+\S+(\s+-j\s+.*)?"

        result: bool = re.match(pattern, input_str, re.IGNORECASE) is not None

        return result

    def execute(self, session: AbstractChatSession, args: List[str] = None):

        # 参数解析增强
        parser = argparse.ArgumentParser(prog=self.name, add_help=False)

        # 添加文件参数
        parser.add_argument("-f", "--file", type=str, help="从文件读取内容")
        # 添加文本块序号参数
        parser.add_argument("-t", "--text_block_seq", type=str, help="从 text_block 中获取相应序号的文本进行发送")

        # 添加JSON文件或文件夹参数
        parser.add_argument("-j", "--json", type=str, help="从JSON文件或文件夹中加载数据用于合成模板")

        try:
            # 解析传入的参数
            parsed_args = parser.parse_args(args) if args else argparse.Namespace()
        except SystemExit:  # 防止argparse自动退出
            print("参数解析错误，请检查输入")
            return

        content: str

        # 如果只有一个参数，加载文件的内容，直接发送
        if 'file' in parsed_args and parsed_args.file is not None:
            with open(args[1], "r", encoding="utf-8") as f:
                content = f.read()
                print(f"(1/1)发送内容：{content}")

        # 如果没有指定 file 文件
        elif 'text_block_seq' in parsed_args and parsed_args.text_block_seq is not None:

            if ',' not in parsed_args.text_block_seq:
                # 获取 text_block_seq 参数
                seq = int(parsed_args.text_block_seq)
                # 获取 text_block 中相应序号的文本
                content = session.command_context['text_blocks'][seq]
            else:
                # 获取 text_block_seq 参数
                seqs: [] = parsed_args.text_block_seq.split(',')

                # 获取 text_block 中相应序号的文本，每段文本间使用 '\n' 分隔
                content = '\n'.join([session.command_context['text_blocks'][int(seq)] for seq in seqs])
        else:
            print("参数错误，请输入正确的参数")
            print("第一个参数为模板文件，第二个参数可以是文件夹，则加载其中所有 .json，可以是文件被直接指定")
            return

        # 如果没有出现 --json 选项，则直接发送
        if 'json' in parsed_args and parsed_args.json is None:
            print(f"直接发送内容：{content}")
            if content is not None:
                msg: HumanMessage = HumanMessage(content)
                session._get_response(msg)

        # 如果出现了 json 参数，并且 json 是指向一个文件
        elif 'json' in parsed_args and parsed_args.json is not None and os.path.isfile(parsed_args.json):

            # 读取文件内容，作为模板
            # 读取第二个参数，作为文件路径
            with open(parsed_args.json, "r", encoding="utf-8") as f:
                data = f.read()
                json_data = json.loads(data)
                # 读取文件内容，作为数据
                # 合成模板和数据
                for key in json_data:
                    content = content.replace("${" + key +"}", json_data[key])

                print(f"发送模板化内容：{content}")

                msg: HumanMessage = HumanMessage(content)
                session._get_response(msg)
        # 如果两个参数，且第二个参数文件夹，则加载里面所有的 json 文件，并分别访问
        elif 'json' in parsed_args and os.path.isdir(args[1]):

            for file in os.listdir(args[1]):
                if str.lower(file).endswith(".json"):
                    with open(os.path.join(args[1], file), "r", encoding="utf-8") as f:
                        data = f.read()
                        json_data = json.loads(data)
                        # 读取文件内容，作为数据
                        # 合成模板和数据

                        # 需要保留content 不变，克隆 content 的值到 this_content
                        this_content = content

                        for key in json_data:
                            this_content = this_content.replace("${" + key +"}", json_data[key])
                        print(f"{file} 发送内容：{this_content}")
                        msg: HumanMessage = HumanMessage(this_content)
                        session._get_response(msg)
        else:
            print("参数错误，请输入正确的参数")
            print("第一个参数为模板文件，第二个参数可以是文件夹，则加载其中所有 .json，可以是文件被直接加载成json")
            return


