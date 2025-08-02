from operator import itemgetter


def extract_list_from_index(source_list: list, index_list: list) -> list:
    return list((itemgetter(*index_list)(source_list)))


def list_to_n_group(list_to_group: list, n: int = 3) -> list:
    length = len(list_to_group)
    remainder = length % n
    if remainder == 0:
        step = length // n
    else:
        step = length // n + 1
    return [list_to_group[i:i + step] for i in range(0, len(list_to_group), step)]