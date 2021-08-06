import os
import getopt
import sys
import torch
from fake_loader import attempt_load


def export_onnx(model, img, f, opset):
    # ONNX model export
    prefix = 'ONNX:'
    try:
        import onnx
        print(f'\n{prefix} starting export with onnx {onnx.__version__}...')
        torch.onnx.export(model, img, f, verbose=False, opset_version=opset,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=['images'],
                          output_names=['output'])

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        # print(onnx.helper.printable_graph(model_onnx.graph))  # print

        print(f'{prefix} export success, saved as {f}')
        print(f"{prefix} run --dynamic ONNX model inference with detect.py: 'python detect.py --weights {f}'")
    except Exception as e:
        print(f'{prefix} export failure: {e}')


def run(model, output_path):
    model = model.to('cpu')
    img = torch.zeros(1, 3, 288, 480).to('cpu')
    # img = torch.zeros(1, 3, 234, 416).to('cpu')

    for _ in range(2):
        model(img)  # dry runs

    export_onnx(model, img, output_path, opset=9)


def main(argv):
    try:
        if not os.path.exists('build'):
            os.mkdir('build')
        opts, args = getopt.getopt(argv, "hi:o:", ["ifile=", "ofile="])
    except getopt.GetoptError:
        print('test.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    model = attempt_load(inputfile, 'cpu')
    run(model.dsp(), outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
