# TODO: find & implement state-of-the-art GAN algorithm
# TODO: create a video reader that can start from any frame
# TODO: create argparse mechanism
# TODO: make sure channels are sorted old2new, i.e. [s_tmk,..., s_t]
# TODO: progressive training: start with a low resolution image and a single conv layer. Gradually double the spatial resolution and add more conv layers.
# TODO: find out why din output is zero
# TODO: maximize log D instead of minimize log(1-D)
# TODO: flip labels when training G and D ???
# TODO: pure minibatches: jaut expert or just fake
# TODO: avoid sparse activations: use leaky Relu
# TODO: avoid max pooling. Use average pooling or conv2d + stride
# TODO: upsample using pixelShuffle or ConvTranspose2d + stride
# TODO: occasionaly sample from ER. Mainly present online examples