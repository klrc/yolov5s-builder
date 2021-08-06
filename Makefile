RUNS_DIR := runs
BUILD_DIR := build
MODEL_NAME := yolov5s
WANDB_DIR := wandb

clean:
	@- $(RM) -rf $(RUNS_DIR)
	@- $(RM) -rf $(BUILD_DIR)
	@- $(RM) -rf $(WANDB_DIR)

all:
	@- cd build; python -m onnxsim $(MODEL_NAME).onnx $(MODEL_NAME)-sim.onnx
	@- cd src/onnx2caffe; python convertCaffe.py ../../build/$(MODEL_NAME)-sim.onnx ../../build/$(MODEL_NAME).prototxt ../../build/$(MODEL_NAME).caffemodel
