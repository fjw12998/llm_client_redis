from configparser import ConfigParser

from cmd_chat import ChatSession
from cmd_chat.cmd_templates import SendTemplateCommand
from cmd_chat.cmd_templates import ShowTextBlockCommand
from cmd_chat.cmd_templates import TextBlockCommand


def main():

    config_path: str = "../../config/config.ini"

    configparser: ConfigParser = ConfigParser()
    configparser.read(config_path, encoding="utf-8")

    model: str = configparser['local_llm']['local_llm_id']

    chat = ChatSession(model=model,
                       llm_json_path="../../config/llm_resources.json",
                       config_path=config_path,
                       max_history=8)  # 保存最近4轮对话

    chat.command_registry.register(SendTemplateCommand())
    chat.command_registry.register(TextBlockCommand())
    chat.command_registry.register(ShowTextBlockCommand())

    chat.start()
    pass

if __name__ == '__main__':
    main()