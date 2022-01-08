from oct_Utils import *


if __name__ == '__main__':
    tasks = get_all_fdas(cache=False)
    print(len(tasks))
    set_seed(0)
    random.shuffle(tasks)
    extract_volume_mp(tasks)
    extract_volume_info_mp(tasks)

    tasks = get_all_volumes(cache=False)
    print(len(tasks))
    set_seed(0)
    random.shuffle(tasks)
    # extract_cp_mp(tasks, size=300)
    # extract_cp_mp(tasks, size=380)
    extract_cp_mp(tasks, size=600)
