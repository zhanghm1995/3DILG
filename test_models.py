'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-10-20 09:04:47
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''
'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-10-18 21:14:23
Email: haimingzhang@link.cuhk.edu.cn
Description: Test the models.
'''

import torch

from modeling_vqvae import Autoencoder

model = Autoencoder(N=512, K=1024, M=2048)

B = 5

surface = torch.rand(B, 2048, 3)
points = torch.rand(B, 4096, 3)
surface = surface * 2.0 - 1
points = points * 2.0 - 1

print(surface.min(), surface.max())
print(points.min(), points.max())


pred = model(surface, points)
logits, z_e_x, z_q_x, sigma, loss_vq, perplexity = pred
print(logits.shape, logits.min(), logits.max())
