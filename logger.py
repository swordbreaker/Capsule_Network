import time
import os


class Logger(object):
    def __init__(self, name: str):
        self.path = f"results/{name}/"
        self.str_time = f"{time.strftime('%Y_%m_%d_%H_%M_%S')}"
        self.file_path = f"{self.path}{self.str_time}.txt"
        self.file = None

        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def __enter__(self):
        self.file = open(self.file_path, 'w+')
        return self

    def __exit__(self, type, value, traceback):
        self.file.close()

    def log(self, msg = ""):
        print(msg, file=self.file)

    def log_and_print(self, msg = ""):
        print(msg, file=self.file)
        print(msg)

    def get_log_path(self, name:str, ext: str) -> str:
        return self.path + name + "_" + self.str_time + ext
