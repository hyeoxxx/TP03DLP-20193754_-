import datasets
import torch, platform

import transformers
print("Torch :", torch.__version__)
print("CUDA?  :", torch.cuda.is_available(), "| cuda libs:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU    :", torch.cuda.get_device_name(0))
    
print(torch.__version__)
print(transformers.__version__)
print(datasets.__version__)