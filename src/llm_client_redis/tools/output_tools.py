from typing import Union


class OutputTools:

    @staticmethod
    def remove_think(value: Union[str, dict]) -> str:
        """
        对输入的参数字符串进行移除 <think>...</think>标签的内容
        :param value: 输入的参数字符串
        :return: 移除 <think>...</think>标签后的字符串
        """

        result: str

        # 判断输入参数是否为字符串类型
        if isinstance(value, str):
            result = value
            # 查找</think>标签的结束位置
            think_idx: int = value.find('</think>') + len('</think>')
        # 判断输入参数是否为字典类型
        elif isinstance(value, dict):
            result = value['answer']
            # 查找</think>标签的结束位置
            think_idx: int = result.find('</think>') + len('</think>')
        else:
            # 如果输入参数既不是字符串也不是字典，则抛出类型错误
            raise TypeError("输入的参数类型不正确，只接受字符串或字典类型")

        # 如果找到了</think>标签，则返回标签后的内容
        if think_idx >= 0:
            return result[think_idx:].strip()
        # 如果没有找到</think>标签，但结果字符串不为空，则返回结果字符串
        elif result:
            return result.strip()
        # 如果结果字符串为空，则直接返回结果字符串
        else:
            return result

    @staticmethod    
    def only_json(value: Union[str, dict]) -> str:

        result: str = value

        if result.startswith("```json"):
            result = result[len("```json"):].strip()
        
        if result.startswith("```"):
            result = result[len("```"):].strip()
        
        if result.endswith("```"):
            result = result[:-len("```")].strip()

        return result
