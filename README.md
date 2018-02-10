DCGAN created based on https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_tensorflow.py . The current used learning 
rate in this code should give good clothes pictures after 500000 number of batches run. Only the learning rate has been tuned so far as it 
gave a stable loss for both the discriminator and the generator continusly. If you want to have a stable loss, I suggest to go for 10e-5 for
the discriminator and 10e-4 for the generator, if the equilibrium still collapses, got for 10e-6 and 10e-4 respectively.
TODO: tune parameters/architecture of the discriminator and increase the learning rate so not have to wait for very long for good looking pictures.
