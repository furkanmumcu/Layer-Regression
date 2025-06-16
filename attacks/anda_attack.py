import sys
import argparse
import os.path
from anda_utils import *
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn import functional as F
import torch
import math


def attack(x, y, model, num_iter=10, sample=False):
	adv_inputs, match, match5 = _attack(x, y, model, num_iter, sample)

	exit_flag = 0
	while match == 1:
		print(exit_flag)
		if exit_flag == 5:
			break
		adv_inputs, match, match5 = _attack(adv_inputs, y, model, num_iter, sample)
		exit_flag = exit_flag + 1

	return adv_inputs

def _attack(x, y, model, num_iter=10, sample=False):
	#x = x.cuda()
	#y = y.cuda()
	#model = model.cuda()

	eps = 8 / 255
	alpha = eps / num_iter
	thetas = get_thetas(int(math.sqrt(25)), -0.3, 0.3)

	min_x = x - eps
	max_x = x + eps


	n_ens = thetas.shape[0]
	xt = x.clone()
	my_dim = x.shape[2]
	device = x.device
	anda = ANDA(data_shape=(1, 3, my_dim, my_dim), device=torch.device(device))
	with torch.enable_grad():
		for i in range(num_iter):

			xt_batch = xt.repeat(n_ens, 1, 1, 1)
			xt_batch.requires_grad = True
			aug_xt_batch = translation(thetas, xt_batch)
			ys = y.repeat(xt_batch.shape[0])
			outputs = model(aug_xt_batch)
			if outputs.ndim == 1:
				outputs = outputs.unsqueeze(0)
			loss = F.cross_entropy(outputs, ys, reduction="sum")
			if model.default_cfg['architecture'] == 'levit_conv_192':
				loss.backward(retain_graph=True)
			else:
				loss.backward()
			new_grad = xt_batch.grad

			anda.collect_model(new_grad)
			sample_noise = anda.noise_mean

			if sample and i == num_iter - 1:
				sample_noises = anda.sample(n_sample=1, scale=1)
				sample_xt = alpha * sample_noises.squeeze().sign() + xt
				sample_xt = torch.clamp(sample_xt, 0.0, 1.0).detach()
				sample_xt = torch.max(torch.min(sample_xt, max_x), min_x).detach()

			xt = xt + alpha * sample_noise.sign()
			xt = torch.clamp(xt, 0.0, 1.0).detach()
			xt = torch.max(torch.min(xt, max_x), min_x).detach()

	if sample:
		adv = sample_xt.detach().clone()
	else:
		adv = xt.detach().clone()

	with torch.no_grad():
		output = model(adv)
	pred_top1 = output.topk(k=1, largest=True).indices
	pred_top5 = output.topk(k=5, largest=True).indices
	if pred_top1.dim() >= 2:
		pred_top1 = pred_top1.squeeze()
	return adv, (pred_top1 == y).sum().item(), \
		(pred_top5 == y.unsqueeze(dim=1).expand(-1, 5)).sum().item()
