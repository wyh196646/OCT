from typing import *
from Utils import *
import accimage
from torchvision.transforms.functional import to_tensor


def run(tasks, config: Dict):
    proc_num, tasks = tasks
    func = eval(config['func_name'])
    results = []
    with Benchmark(config['func_name'], print=proc_num == 0) as t:
        for i in range(len(tasks)):
            if i % (len(tasks) // 10 + 1) == 0:
                t.print_elapsed(f'{i}/{len(tasks)}, {i / len(tasks) * 100:.2f}%')
            task = tasks[i]
            results.append(func(task, config))
    return results


def printable_dict(x):
    result = {}
    for k, v in x.items():
        result[str(k)] = str(v)
    return result


def run_mp(tasks: List, config: Dict):
    n_proc = config['n_proc']

    print(f"Run {config['func_name']}, config = \n{json.dumps(printable_dict(config), indent=4, ensure_ascii=False)}")
    return run_multi_process(tasks, n_proc, partial(run, config=config), with_proc_num=True)


def image_resize(filepath: str, config: Dict):
    input_root = config['input_root']
    output_root = config['output_root']
    output_size = config['output_size']
    output_suffix = config['output_suffix']

    input_path = str(input_root / filepath)
    output_path = str((output_root / filepath).with_suffix(output_suffix))

    img = Image.open(input_path)
    img = img.resize(output_size, Image.ANTIALIAS)
    ensure_file(output_path)
    img.save(output_path)


def video_to_frames(filepath: str, config: Dict):
    input_root = config['input_root']
    output_root = config['output_root']
    output_size = config['output_size']
    output_suffix = config['output_suffix']

    input_path = str(input_root / filepath)
    output_path = (output_root / filepath).with_suffix('')
    ensure_path(output_path)

    vidcap = cv2.VideoCapture(input_path)
    success,image = vidcap.read()
    count = 0
    while success:
        image = cv2.resize(image, dsize=output_size, interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(output_path / f'frame_{count}{output_suffix}'), image)
        success,image = vidcap.read()
        count += 1


def get_mean_std(filepath: str, config: Dict):
    input_root = config['input_root']
    input_path = str(input_root / filepath)
    img = to_tensor(accimage.Image(input_path))
    img = img.view(3, -1)
    means = img.mean(dim=-1)
    stds = img.std(dim=-1)
    result = torch.cat([means, stds], dim=0).numpy()
    return result


def convert(config: Dict):
    input_root = config['input_root']
    input_suffix = config['input_suffix']
    if not config['cached']:
        tasks = []
        for file in tqdm(input_root.rglob(f'*{input_suffix}')):
            tasks.append(str(file.relative_to(input_root)))
        pkl_dump(tasks, 'tmp/tasks.pkl')
    else:
        tasks = pkl_load('tmp/tasks.pkl')
    print(f'Collected all files: {len(tasks)}, example: {tasks[0]}')
    return run_mp(tasks, config)


if __name__ == '__main__':
    img_root = Path('/home/octusr2/projects/data_fast/proceeded/cp_projection/')
    img_root_600 = img_root / '600'
    img_root_512 = img_root / '512'
    img_root_256 = img_root / '256'

    config = {
        'input_root': img_root_600,
        'output_root': img_root_256,
        'func_name': 'image_resize',
        'output_size': (256, 256),
        'input_suffix': '.jpg',
        'output_suffix': '.jpg',
        'n_proc': 80,
        'cached': False
    }

    convert(config)
