import torch
import resnet

#modelPath = "../models/facenet/20220213-154936_resnet18_full/resnet18_full_30.pt"
#modelPath = "../models/facenet/20220214-131718_resnet18_full/resnet18_full_0.pt"
modelPath = "../models/facenet/20220214-133113_resnet18_full/resnet18_full_0.pt"
model = resnet.resnet18(binarized=False)
model.load_state_dict(torch.load(modelPath))
model.eval()

dummy_input = torch.rand(1, 3, 56, 56)
output = model.forward(dummy_input)
output_magnitude = torch.linalg.norm(output)

print(output)
print(output_magnitude)
