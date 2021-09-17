from pathlib import Path
import os

from oirpy.oirreader import Oirreader

def oirloader(filepath):

    if isinstance(filepath, str):
        filepath = Path(filepath)
    report_file = filepath.joinpath(str(filepath).replace(".oir", "_error.txt"))

    try:
        oir_image = Oirreader(filepath)
        channels = oir_image.get_meta()["channel_names"]
        stack = oir_image.get_stack()
    except:
        with open(report_file, "a+") as f:
            f.write("Loading error")
        return None
    return stack, channels

def find_oir_files(directory):

    all_dirs = []
    for dirName, subdirList, fileList in os.walk(directory):
        if len(list(Path(dirName).glob('*.oir'))) > 0:
            all_dirs.append(dirName)

    return all_dirs