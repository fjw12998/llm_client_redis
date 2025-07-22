# file: show_text_block_cmd.py
import re

from ..command_def import Command, AbstractChatSession
import argparse
from typing import List

class ShowTextBlockCommand(Command):
    def __init__(self):
        super().__init__(
            name="show",
            aliases=[],
            description="显示文本区块内容（-t [序号] 显示指定区块，不带序号则列出所有）"
        )

    def match(self, input_str: str, session: AbstractChatSession) -> bool:

        pattern: str = r"^(show)(\s+-t\s*(\d*))?$"
        result: bool = re.match(pattern, input_str.strip(), re.IGNORECASE) is not None

        return result


    def execute(self, session: AbstractChatSession, args: List[str] = None):
        parser = argparse.ArgumentParser(prog="show", add_help=False)
        parser.add_argument("-t", "--text-block", type=int, nargs='?', const=-1, 
                          help="指定要显示的文本区块序号")
        try:
            parsed_args = parser.parse_args(args) if args else argparse.Namespace()
        except SystemExit:
            parsed_args = argparse.Namespace(text_block=None)

        text_blocks = session.command_context.get('text_blocks', [])

        # 显示指定区块内容
        if parsed_args.text_block is not None:
            block_id = parsed_args.text_block
            if block_id < 0:  # 处理 show -t 不带参数的情况
                self._list_all_blocks(text_blocks)
                return

            # 需要判断 text_blocks 的第 block_id 的索引的元素是否有内容
            if len(text_blocks) <= block_id or text_blocks[block_id] is None:
                print(f"❌ 文本区块 {block_id} 不存在")
                return

            content = text_blocks[block_id]
            print(f"\n=== 文本区块 {block_id} ===")
            print(content)
            print(f"=== 共 {len(content.splitlines())} 行，{len(content)} 字符 ===\n")

        # 列出所有区块
        else:
            self._list_all_blocks(text_blocks)

    def _list_all_blocks(self, text_blocks: []):
        """
        列出所有文本区块的信息
        :param text_blocks: 文本区块字典
        :return:
        """
        if not text_blocks:
            print("❌ 没有可用的文本区块")
            return

        print("\n=== 文本区块列表 ===")
        for block_id in range(len(text_blocks)):

            content = text_blocks[block_id]

            if content is None:
                continue

            if content.strip():  # 只显示非空内容
                lines = len(content.splitlines())
                chars = len(content)
                print(f"[区块 {block_id}] 行数: {lines:<3} 字符数: {chars}")

    def help(self, session: AbstractChatSession) -> str:
        return "show -t [序号]: 显示文本区块（-t不带序号则列出所有）"
