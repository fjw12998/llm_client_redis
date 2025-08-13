import time
from datetime import datetime


class TimeController:
    """
    时间控制器，用于控制程序在指定时间运行
    """

    @staticmethod
    def time_to_begin(date_time: str, internal: int = 10) -> None:
        """
        判断时间是否到达, 如果到达则跳出循环
        :param date_time: 时间字符串，格式为 '2023-07-01 12:40:00'
        :param internal: 循环间隔时间，默认为 10
        :return: None
        """
        # 设置跳出循环的时间
        end_time = datetime.strptime(date_time, '%Y-%m-%d %H:%M:%S')

        while True:
            # 获取当前时间
            current_time = datetime.now()

            # 如果当前时间超过 12:40，跳出循环
            if current_time >= end_time:
                print(f"当前时间已超过 {date_time}，跳出计时器循环")
                break

            # 显示当前时间
            print(f"当前时间: {current_time}，未达目标：{date_time}，仍需等待")

            # 等待 10 秒
            time.sleep(internal)

    @staticmethod
    def time_to_run_or_wait(begin_time_str: str, end_time_str: str) -> bool:
        """
        判断当前系统的时间是否在 begin_time_str 和 end_time_str 之间，如果是则运行，否则等待
        :param begin_time_str: 开始时间字符串，格式为 '12:40:00'
        :param end_time_str: 结束时间字符串，格式为 '12:50:00'
        :return: True 表示在时间范围内，False 表示不在时间范围内
        """
        # 以 '%H:%M:%S' 获取当前时间
        current_time = datetime.now().strftime('%H:%M:%S')

        # 判断当前时间是否在开始时间和结束时间之间
        if begin_time_str <= current_time <= end_time_str:
            return True

        # 处理跨越午夜的情况，例如开始时间为 '22:00:00'，结束时间为 '02:00:00'
        elif begin_time_str >= end_time_str and (current_time >= begin_time_str or current_time <= end_time_str):
            return True
        else:
        # 当前时间不在指定范围内，等待指定的时间后再次判断
            return False