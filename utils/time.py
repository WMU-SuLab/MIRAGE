from datetime import datetime


def datetime_now() -> datetime:
    return datetime.now()


def datetime_now_str() -> str:
    return datetime_now().strftime('%Y%m%d%H%M%S')


def datetime_now_str_multi_train() -> str:
    return datetime_now().strftime('%Y%m%d%H%M')
