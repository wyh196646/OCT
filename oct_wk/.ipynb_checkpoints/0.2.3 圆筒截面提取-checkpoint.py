from oct_Utils import *


if __name__ == '__main__':
    tasks = get_all_volumes(cache=True)
    extract_cp_mp(tasks)
