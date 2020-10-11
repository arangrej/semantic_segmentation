import os
import cv2
from optparse import OptionParser


def resize(input_base_path='', output_base_path='', height=224, width=224):

    dirs = ['training', 'testing']
    sub_dirs = ['input', 'output']

    for dir in dirs:

        if not os.path.exists(os.path.join(output_base_path, dir)):
            os.makedirs(os.path.join(output_base_path, dir))

        for sub_dir in sub_dirs:

            if not os.path.exists(os.path.join(output_base_path, dir, sub_dir)):
                os.makedirs(os.path.join(output_base_path, dir, sub_dir))

            names = os.listdir(os.path.join(input_base_path, dir, sub_dir))
            for name in names:

                if not os.path.exists(os.path.join(input_base_path, dir, 'output', name)):
                    continue

                img = cv2.imread(os.path.join(input_base_path, dir, sub_dir, name), cv2.IMREAD_UNCHANGED)
                dim = (width, height)
                resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(output_base_path, dir, sub_dir, name), resized)


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("-i", "--input_base_path", action='store', type='str', dest='input_base_path')
    parser.add_option("-o", "--output_base_path", action='store', type='str', dest='output_base_path')
    parser.add_option("-y", "--height", action='store', type='int', dest='height', default=224)
    parser.add_option("-x", "--width", action='store', type='int', dest='width', default=224)

    options, _ = parser.parse_args()
    resize(options.input_base_path, options.output_base_path, options.height, options.width)
