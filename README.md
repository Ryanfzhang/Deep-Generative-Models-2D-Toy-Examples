# Deep-Generative-Models-2D-Toy-Examples

This is an comparison of different deep generative model for 2D samples generation, including [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239), [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661) and [Variational Auto-Encoder](https://arxiv.org/pdf/1312.6114.pdf). 

A simple pytorch network learns to predict the noise component in a data sample. This is then used in a DDPM sampler to generate new samples from the distribution.
