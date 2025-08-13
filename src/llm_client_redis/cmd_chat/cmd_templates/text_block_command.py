from ..command_def import Command, AbstractChatSession

from typing import List
import re
import argparse
import requests


class TextBlockCommand(Command):
    def __init__(self, max_text_seq: int = 9):

        self.default_task_name = "text:"
        
        """
        :param max_text_seq: 文本区块ID (0-9)
        """
        super().__init__(
            name=self.default_task_name,  # 命令名称格式：Text0-Text[max_text_seq]
            aliases=[f"T"],  # 添加简写别名 T0-T9
            description=f"存储多行文本到区块从0到{max_text_seq}（支持 -e 结束符参数 -f 读取本地文件路径参数 -u 读取url资源的参数）"
        )
        self.max_text_seq = max_text_seq
        self.default_end_marker = "END"  # 默认结束标记
        self.block_id: int = 0  # 当前区块ID

    def match(self, input_str: str, session: AbstractChatSession) -> bool:
        # 使用正则表达式匹配命令格式
        # 例如：Text0, Text1, Text2, ..., Text9
        pattern = rf"^(text[0-{self.max_text_seq}]:?(\s*|\s+).*)$"

        cmd_name: str = input_str.strip()

        # 如果输入字符串匹配正则表达式，则返回True，否则返回False
        result: bool = re.match(pattern, cmd_name, re.IGNORECASE) is not None

        if result:
            # 如果匹配成功，提取区块ID
            self.block_id = int(re.search(r"text(\d+)", input_str.strip()).group(1))

            self.name = f"Text{self.block_id}"  # 更新命令名称

        return result

    def execute(self, session: AbstractChatSession, args: List[str] = None):

        end_marker: str = self.default_end_marker

        # 参数解析增强
        parser = argparse.ArgumentParser(prog=self.name, add_help=False)

        # 添加结束符号参数
        parser.add_argument("-e", "--end", type=str, default=self.default_end_marker,
                            help="指定文本缓存的结束符号")
        # 添加文件参数
        parser.add_argument("-f", "--file", type=str, help="从文件读取内容")
        # 添加URL参数
        parser.add_argument("-u", "--url", type=str, help="从URL读取内容")

        try:
            # 解析传入的参数
            parsed_args = parser.parse_args(args) if args else argparse.Namespace(end=self.default_end_marker)
            # 更新结束标记符
            end_marker = parsed_args.end
        except SystemExit:  # 防止argparse自动退出
            print(f"参数解析错误，使用默认结束符：{self.default_end_marker}")
            # 设置默认参数
            parsed_args = argparse.Namespace(end=self.default_end_marker, file=None, url=None)

        lines = []
        # 文件/URL输入优先
        if 'file' in parsed_args:
            try:
            # 从文件读取内容
                with open(parsed_args.file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"文件读取失败: {str(e)}")
        elif 'url' in parsed_args:
            try:

            # 从URL读取内容
                resp = requests.get(parsed_args.url)
                resp.raise_for_status()
                lines = resp.text.splitlines()
            except Exception as e:
                print(f"URL获取失败: {str(e)}")

        # 交互式输入（当没有文件/URL参数时）
        if not lines:
            print(f"\n正在输入到区块{self.block_id}（输入 '{end_marker}' 或 Ctrl+D 结束）:")
            while True:
                try:
                    prompt = "> " if not lines else "... "
                    line = input(prompt)
                    if line.strip() == end_marker:
                        break
                    lines.append(line)
                except (KeyboardInterrupt, EOFError):
                    print(f"\n输入结束，已保存{len(lines)}行")
                    break

        # 存储数据
        if 'text_blocks' not in session.command_context:
            session.command_context['text_blocks'] = []

            # 确保列表长度足够
        if len(session.command_context['text_blocks']) <= self.block_id:
            session.command_context['text_blocks'].extend(
                [None] * (self.block_id + 1 - len(session.command_context['text_blocks'])))

        session.command_context['text_blocks'][self.block_id] = "\n".join(lines)
        print(f"✅ 已保存{len(lines)}行内容到文本区块{self.block_id}")

    def help(self, session: AbstractChatSession) -> str:
        return f"{self.default_task_name} 多行文本输入到区块[0-{self.max_text_seq}]，-e指定结束符,默认END -f指定文件路径 -u指定URL路径"
