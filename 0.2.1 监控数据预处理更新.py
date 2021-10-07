from oct_Utils import *
from datetime import datetime
import time


if __name__ == '__main__':
    now = datetime.now()
    while True:
        while now.hour != 5 or now.minute != 0:
            now = datetime.now()
            time.sleep(1)
            continue
        print(f'Start preprocessing at {now}')
        tasks = get_all_fdas(cache=False)
        print(len(tasks))
        set_seed(0)
        random.shuffle(tasks)
        extract_volume_mp(tasks)
        extract_volume_info_mp(tasks)

        tasks = get_all_volumes(cache=False)
        extract_cp_mp(tasks, size=380)
        print(f'Finish preprocessing at {now}')

        delete_invalid_files()
    