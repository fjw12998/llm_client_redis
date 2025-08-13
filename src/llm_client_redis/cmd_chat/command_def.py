import traceback
import shlex
from abc import ABC, abstractmethod
from typing import List, Optional
from llm_client_redis import LLMClientRedis
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage


class AbstractChatSession(ABC):
    def __init__(self, model: str,
                 llm_json_path: str,
                 config_path: str,
                 save_output_path: str = "./",
                 max_history=6):
        self.history: List[BaseMessage] = []
        self.max_history = max_history
        self.model = model
        self.llm_redis: LLMClientRedis = LLMClientRedis(llm_json_path=llm_json_path, config_path=config_path)
        self.save_output_path = save_output_path
        self.user_input = ""  # 新增字段存储当前输入

        # 初始化命令系统
        self.command_registry = CommandRegistry(session=self)
        self.init_commands()

        # 初始化一下命令的上下文对象，用于为命令保存缓存上下文的位置
        self.command_context: dict = {}

    @abstractmethod
    def init_commands(self):
        """初始化所有命令"""
        # # 注册基础命令
        # self.command_registry.register(ExitCommand())
        # self.command_registry.register(HistoryCommand())
        # self.command_registry.register(DeleteHistoryCommand())
        # self.command_registry.register(ClearHistoryCommand())
        # self.command_registry.register(HelpCommand())
        # self.command_registry.register(ListModelsCommand())
        # self.command_registry.register(CurrentModelCommand())
        # self.command_registry.register(SummaryCommand())
        # self.command_registry.register(SaveCommand())
        #
        # # 动态注册模型切换命令
        # models = self.llm_redis.llm_manager.list_llm_def()
        # for idx, model in enumerate(models):
        #     self.command_registry.register(ChangeModelCommand(idx, model))
        pass

    def start(self):
        self.show_welcome()
        while True:
            try:
                self.user_input = input("\n请你发言: ").strip()
                if not self.user_input:
                    continue

                # 命令处理
                if cmd := self.command_registry.find_command(self.user_input):

                    # 将 self.user_input 除去命令的部分，生成参数数组
                    args_str = self.user_input[len(cmd.name):].strip()
                    args = shlex.split(args_str, posix=True) if args_str else None

                    cmd.execute(session=self, args=args)
                    continue

                # 非命令处理流程
                human_input = HumanMessage(self.user_input)
                self.get_response(human_input)

            except KeyboardInterrupt:
                print("\n\n操作已经被取消。请输入 'exit' 或 'quit' 进行退出。")
            except Exception as e:
                print(f"\n[系统错误] {str(e)}")
                print(f"\n[系统错误] {traceback.format_exc()}")

    def show_welcome(self):
        """动态生成帮助信息"""
        print("""\n
====================================
对话助手 (输入命令控制交互)
支持命令：""")
        # 通过命令系统生成帮助信息
        for cmd in self.command_registry.list_commands():
            print(f"    {cmd.help(session=self)}")
        # 此处应遍历所有注册命令生成帮助信息...
        print("""\n====================================
        """)

    def get_response(self, user_input: HumanMessage) -> None:
        """获取模型响应并管理历史记录"""
        # 添加用户输入到历史（带角色标识）
        self.history.append(user_input)

        print(f"\n系统[{self.model}]: ", end="")

        answer_text: str = ""

        # 调用大模型接口
        for answer_chunk in self.llm_redis.request_stream(messages=self.history.copy(), model=self.model):
            answer_text += answer_chunk
            print(answer_chunk, end="", flush=True)

        # 添加AI响应到历史
        self.history.append(SystemMessage(answer_text))

        # 保持历史记录不超过限制
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        return None

    @abstractmethod
    def show_history(self) -> str:
        """
        定义一个私有方法_show_history，用于显示对话历史，返回类型为字符串

        :return:
        """
        pass

    @abstractmethod
    def del_history(self, idx: int):
        """
        定义一个私有方法del_history，用于删除指定索引的历史记录
        :param idx: 要删除的历史记录的索引
        :return: None
        """
        pass

    @abstractmethod
    def summary_history(self):
        pass

    @abstractmethod
    def save_history(self):
        pass

    @abstractmethod
    def list_models(self):
        pass

    @abstractmethod
    def show_model(self):
        pass

    @abstractmethod
    def clear_history(self):
        pass


class Command(ABC):
    """命令基类"""
    def __init__(self, name: str, aliases: List[str], description: str):
        """初始化命令对象
        :param name
        """

        # 初始化命令对象，设置命令名称、别名和描述
        self.name = name  # 设置命令的名称
        self.aliases = aliases  # 设置命令的别名列表
        self.description = description  # 设置命令的描述信息

    def match(self, input_str: str, session: AbstractChatSession) -> bool:
        """判断是否匹配命令"""
        # 去除输入字符串的前后空格并转换为小写
        clean_input = input_str.strip().lower()
        # 判断清理后的输入是否在命令名称的小写形式或别名的小写形式中
        return clean_input in [self.name.lower()] + [a.lower() for a in self.aliases]

    @abstractmethod
    def execute(self, session: AbstractChatSession, args: List[str] = None):
        """执行命令"""
        # 抽象方法，子类必须实现此方法
        pass

    def help(self, session: AbstractChatSession) -> str:
        """帮助信息"""
        # 如果有别名，将别名用斜杠连接成一个字符串，否则为空字符串
        aliases = "/".join(self.aliases) if self.aliases else ""
        # 返回命令的帮助信息，格式为“命令名称(别名): 描述”
        return f"{self.name}{f'({aliases})' if aliases else ''}: {self.description}"


class CommandRegistry:
    """命令注册中心"""
    def __init__(self, session: AbstractChatSession):
        self.commands: List[Command] = []
        self.session = session

    def register(self, command: Command):
        self.commands.append(command)

    def list_commands(self) -> List[Command]:
        return self.commands.copy()

    def find_command(self, input_str: str) -> Optional[Command]:
        clean_input = input_str.strip().lower()
        for cmd in self.commands:
            if cmd.match(input_str=clean_input, session=self.session):
                return cmd
        return None


