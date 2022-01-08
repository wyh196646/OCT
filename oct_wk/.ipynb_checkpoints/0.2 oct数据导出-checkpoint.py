from Utils import *
from fda import FDA
from fds import FDS
import sys


def convert(tasks, input_root, output_root):
    proc_num, tasks = tasks
    errors = []
    with Benchmark(f'Converting', print=proc_num == 0) as t:
        for i in range(len(tasks)):
            if i % (len(tasks) // 10) == 0:
                t.print_elapsed(f'{i}/{len(tasks)}, {i / len(tasks) * 100:.2f}%')
            task = tasks[i]
            input_path = input_root / task
            try:
                if input_path.suffix == '.fda':
                    fda = FDA(str(input_path))
                    oct_volume = fda.read_oct_volume()
                    result = np.stack(oct_volume.volume).astype(np.uint8)
                    ensure_file(str(output_root / task.with_suffix('.npy')))
                    np.save(str(output_root / task.with_suffix('.npy')), result)
                else:
                    fds = FDS(str(input_path))
                    oct_volume = fds.read_oct_volume()
                    result = (np.stack(oct_volume.volume)/256).astype(np.uint8)
                    ensure_file(str(output_root / task.with_suffix('.npy')))
                    np.save(str(output_root / task.with_suffix('.npy')), result)
            except Exception as err:
                errors.append((task, err))
                print(f'Error with file {str(task)}: {err}')
    return errors

                    
if __name__ == '__main__':
    input_root = Path('/home/octusr2/projects/data_fast/data')
    output_root = Path('/home/octusr2/projects/data_fast/processed/volume_npy')
    tasks = []
    for file in chain(input_root.rglob('*.fda'), input_root.rglob('*.fds')):
        tasks.append(file.relative_to(input_root))
    set_seed(0)
    random.shuffle(tasks)
    tasks = tasks[len(tasks) // 2:]
    errors = run_multi_process(tasks, 50, partial(convert, input_root=input_root, output_root=output_root), with_proc_num=True)
    errors = reduce(lambda x, y: x + y, errors)
    pkl_dump(errors, 'data/errors_1.pkl')
