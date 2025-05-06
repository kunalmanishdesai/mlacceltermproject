# CSE 524 (494): Machine Learning Accelerated Design Project
Term Project for ML Acceleration

In this repository, you will find code and results for our Machine Learning Acceleration Term Project



```
module load mamba/latest
source activate scicomp
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install nvidia-tensorrt --extra-index-url https://pypi.nvidia.com
pip install torch-tensorrt -f https://github.com/pytorch/TensorRT/releases

To run tensorrt_quant_ops first run 
pip install nvidia-modelopt[all]

ml purge && ml cuda-12.2.1-gcc-12.1.0
```