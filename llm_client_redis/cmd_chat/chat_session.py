import datetime
import os

from langchain_core.messages import HumanMessage, SystemMessage

from llm_client_redis.tools import OutputTools
from .command_def import Command
from .command_def import AbstractChatSession

# ------------------------ 新增命令体系 ------------------------
class ExitCommand(Command):
    """退出命令"""
    def __init__(self):
        super().__init__("exit", ["quit"], "退出程序")

    def execute(self, session: AbstractChatSession, args=None):
        print("再见!")
        exit(0)

class HistoryCommand(Command):
    """显示历史命令"""
    def __init__(self):
        super().__init__("history", [], "显示对话历史")

    def execute(self, session: AbstractChatSession, args=None):
        session._show_history()

class DeleteHistoryCommand(Command):
    """删除历史命令"""
    def __init__(self):
        super().__init__("hdel", [], "删除指定历史记录 hdel[索引]")

    def match(self, input_str: str, session: AbstractChatSession) -> bool:

        if input_str.lower().startswith(self.name):
            idx: int = int(input_str.lower().strip()[len("hdel"):])
            return idx >= 0 and idx < len(session.llm_redis.llm_manager.list_llm_def())
        else:
            return False


    def execute(self, session: AbstractChatSession, args=None):
        idx_str = session.user_input[len(self.name):]
        try:
            idx = int(idx_str)
            session._del_history(idx)
        except ValueError:
            print(f"无效的索引: {idx_str}")

class ClearHistoryCommand(Command):
    """清空历史命令"""
    def __init__(self):
        super().__init__("clear", [], "清空对话历史")

    def execute(self, session: AbstractChatSession, args=None):
        session._clear_history()

class HelpCommand(Command):
    """帮助命令"""
    def __init__(self):
        super().__init__("help", ["?"], "显示帮助信息")

    def execute(self, session: AbstractChatSession, args=None):
        session._show_welcome()

class ListModelsCommand(Command):
    """列出模型命令"""
    def __init__(self):
        super().__init__("list", [], "列出支持的所有模型")

    def execute(self, session: AbstractChatSession, args=None):
        session._list_models()

class CurrentModelCommand(Command):
    """当前模型命令"""
    def __init__(self):
        super().__init__("model", [], "显示当前使用模型")

    def execute(self, session: AbstractChatSession, args=None):
        session._show_model()

class SummaryCommand(Command):
    """摘要命令"""
    def __init__(self):
        super().__init__("summary", [], "生成历史摘要")

    def execute(self, session: AbstractChatSession, args=None):
        session._summary_history()

class SaveCommand(Command):
    """保存命令"""
    def __init__(self):
        super().__init__("save", [], "保存对话历史")

    def execute(self, session: AbstractChatSession, args=None):
        session._save_history()

class ChangeModelCommand(Command):
    """切换模型命令"""
    def __init__(self):
        super().__init__(f"chg", [], f"切换到list 返回的[索引]模型作为当前模型")

    def match(self, input_str: str, session: AbstractChatSession) -> bool:

        if input_str.lower().startswith(self.name):
            idx: int = int(input_str.lower().strip()[len("chg"):])
            return idx >= 0 and idx < len(session.llm_redis.llm_manager.list_llm_def())
        else:
            return False

    def execute(self, session: AbstractChatSession, args=None):

        idx_str = session.user_input[len(self.name):]

        model: str = session.model
        try:
            idx = int(idx_str)
            model = session.llm_redis.llm_manager.list_llm_def()[idx]
        except ValueError:
            print(f"无效的索引: {idx_str}")

        session.model = model
        print(f"当前模型变更为：{session.model}")

# ------------------------ 修改后的ChatSession类 ------------------------
class ChatSession(AbstractChatSession):

    def _init_commands(self):
        """初始化所有命令"""
        # 注册基础命令
        self.command_registry.register(ExitCommand())
        self.command_registry.register(HistoryCommand())
        self.command_registry.register(DeleteHistoryCommand())
        self.command_registry.register(ClearHistoryCommand())
        self.command_registry.register(HelpCommand())
        self.command_registry.register(ListModelsCommand())
        self.command_registry.register(CurrentModelCommand())
        self.command_registry.register(SummaryCommand())
        self.command_registry.register(SaveCommand())
        self.command_registry.register(ChangeModelCommand())

    def _show_history(self) -> str:
        """
        定义一个私有方法_show_history，用于显示对话历史，返回类型为字符串

        :return:
        """
        text: str = ""  # 初始化一个空字符串变量text，用于存储对话历史文本

        text += "\n=== 对话历史 ===\n"  # 向text中添加对话历史开始的分隔符
        for idx, line in enumerate(self.history[-self.max_history:]):
            text += f"{idx}. {line.type}: {line.content}\n"  # 打印历史记录，每条记录前加上序号
        text += "=================\n"

        print(text)
        return text

    def _del_history(self, idx: int):
        """
        定义一个私有方法_del_history，用于删除指定索引的历史记录
        :param idx: 要删除的历史记录的索引
        :return: None
        """
        # 检查索引是否在有效范围内
        if idx < 0 or idx >= len(self.history):
            # 如果索引无效，打印错误信息并返回
            print(f"无效的历史记录索引: {idx}")
            return
        # 删除指定索引的历史记录
        del self.history[idx]
        # 打印删除成功的信息
        print(f"删除历史记录: {idx}")
        # 调用私有方法_show_history显示当前的历史记录
        self._show_history()
        # 返回None，表示方法执行完毕
        return

    def _summary_history(self):
        print("对当前的所有历史记录进行汇总摘要，归纳成一个上下文提示词。")
        self._show_history()
        self.history.append(HumanMessage("对当前的所有历史记录进行汇总摘要，归纳成一个上下文提示词，"
                                         "以便在后续的对话中获得前面话题的核心内容。"))

        print(f"\n系统：", end="")

        answer_text: str = ""
        # 调用大模型接口
        for answer_chunk in self.llm_redis.request_stream(messages=self.history.copy(), model=self.model):
            answer_text += answer_chunk
            print(answer_chunk, end="", flush=True)

        print("摘要替代原上下文")
        self.history = [SystemMessage(OutputTools.remove_think(answer_text))]
        self._show_history()

    def _save_history(self):
        # 调用_show_history方法获取对话历史文本
        text: str = self._show_history()

        # 生成文件的路径，文件名包含当前的时间
        # 使用os.path.join拼接保存路径和文件名
        # 文件名格式为"chat_history_YYYYMMDD_HHMMSS.txt"，其中YYYYMMDD_HHMMSS为当前时间
        file_path: str = os.path.join(self.save_output_path,
                                      f"chat_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        # 使用with语句打开文件，确保文件在操作完成后自动关闭
        # 'w'模式表示写入文件，如果文件不存在则创建文件
        # encoding='utf-8'指定文件编码为UTF-8
        with open(file_path, 'w', encoding='utf-8') as f:
            # 将对话历史文本写入文件
            f.write(text)
        # 打印保存文件的路径
        print(f"保存对话历史，路径：{file_path}")

    def _list_models(self):
        models: [] = self.llm_redis.llm_resources_tools.list_llm_def()

        print(f"\n当前系统支持以下模型：")
        for idx, model in enumerate(models):
            print(f"{idx} - {model}")
        print()

    def _show_model(self):
        print(f"\n当前使用的模型：{self.model}")

    def _clear_history(self):
        self.history = []
        print("对话历史已清空。")


if __name__ == "__main__":
    chat = ChatSession(model="home_qwen3:32b",
                       llm_json_path="../config/llm_resources.json",
                       config_path="../config/config.ini",
                       max_history=8)  # 保存最近4轮对话
    chat.start()