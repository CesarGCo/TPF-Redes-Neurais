import torch

state_dict = torch.load("uncased_L-12_H-768_A-12/pytorch_model.bin", map_location="cpu")

new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith("bert."):
        new_key = key[len("bert."):]
        new_state_dict[new_key] = value

torch.save(new_state_dict, "uncased_L-12_H-768_A-12/pytorch_model_fixed.bin")
print("Conversao concluida! Chaves salvas:", len(new_state_dict))