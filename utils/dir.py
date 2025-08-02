import os


def mk_dir(dir_path: str) -> bool:
    path = str(dir_path).strip()
    if not os.path.exists(path) or not os.path.isdir(path):
        try:
            os.makedirs(path)
            # os.mkdir(path)
        except Exception as e:
            print(str(e))
            return False
    return True
