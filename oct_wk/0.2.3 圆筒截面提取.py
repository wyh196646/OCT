from oct_Utils import *


if __name__ == '__main__':
    tasks = get_all_volumes(cache=False)
    extract_cp_mp(tasks, size=512)
