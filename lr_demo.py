import torch
import numpy as np
import timm
import mlp
import dataloader as dt
import utils

torch.set_printoptions(sci_mode=False)
device = 'cuda'

model = timm.create_model('inception_v3', pretrained=True)
model.eval()
model = model.to(device)

loss_fn = torch.nn.MSELoss()

model_mlp = mlp.MLP(8217, 2048).to(device)
model_mlp_dir = 'lr-models/model_inception.pt'
model_mlp.load_state_dict(torch.load(model_mlp_dir))
model_mlp.eval()

layers = []
for name, module in model.named_modules():
	#print(name) # names of all layers
	if "conv" in name or "head_drop" == name:
		#print(f"Module hook registered for: {name} # names of tracked layers
		module.register_forward_hook(lambda m, i, o: layers.append(o))


test_loader = dt.get_loader(split='test', resize_dim=299, shuffle=False, batch=1)
img_indexes = [0, 500, 1500, 7500, 9500]
for i in img_indexes:
	img, lbl = test_loader.dataset[i]

	img = torch.unsqueeze(img.to(torch.float32).to(device), dim=0)
	lbl = torch.from_numpy(np.array([lbl])).type(torch.long).to(device)

	outputs = model(img)
	_, pre = torch.max(outputs.data, 1)

	# Check if the initial prediction is correct, this should print True
	print(pre == lbl)


	# Get the LR score for clean sample
	f1 = layers[15][:, :3].view(1, 3 * 35 * 35)
	f2 = layers[25][:, :3].view(1, 3 * 35 * 35)
	f3 = layers[35][:, :3].view(1, 3 * 17 * 17)
	f_tot = torch.cat([f1, f2, f3], dim=1)

	o2 = layers[-1]

	mlp_out = model_mlp(f_tot)
	lr_score_clean = loss_fn(mlp_out, o2)

	# Attack and get the LR score for attacked sample
	adv_img = utils.get_attack(img, lbl, model, 'pgd')

	del layers
	layers = []

	outputs_adv = model(adv_img)
	_adv, pre_adv = torch.max(outputs_adv.data, 1)

	# Check if attack successful, this should print False
	print(pre_adv == lbl)

	f1 = layers[15][:, :3].view(1, 3 * 35 * 35)
	f2 = layers[25][:, :3].view(1, 3 * 35 * 35)
	f3 = layers[35][:, :3].view(1, 3 * 17 * 17)
	f_tot = torch.cat([f1, f2, f3], dim=1)

	o2 = layers[-1]

	mlp_out = model_mlp(f_tot)
	lr_score_adv = loss_fn(mlp_out, o2)

	del layers
	layers = []

	print('LR score for clean: {} --- LR score for attacked sample: {}'.format(lr_score_clean.item(), lr_score_adv.item()))
	print()
