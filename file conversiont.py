from flux_module import convert_image
import os
import pathlib

working_dir = pathlib.PureWindowsPath(os.getcwd()).as_posix()

for image in os.listdir(working_dir + '/Images to Process'):
    size_original_stack, size_final_stack = convert_image(relative_image_path=image, channel=1)
    print(f'Size Original Stack: {size_original_stack},\nSize Final Image: {size_final_stack}')

