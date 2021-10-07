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
        tasks = get_all_fdas(cache=True)
        print(len(tasks))
        set_seed(0)
        random.shuffle(tasks)
        extract_volume_info(tasks)
