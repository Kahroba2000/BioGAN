<!-- <img src='imgs/horse2zebra.gif' align="right" width=384> 

<br><br><br>
-->
# BioGAN

A GAN-based image to image translation model for cell biology images. This model is developed on top of cycle GAN model (https://github.com/nanfengpo/CycleGAN-tensorflow-1?msclkid=d1fd27c8adbc11ec9a2f9153bce627cd). 

![temp](https://user-images.githubusercontent.com/45915632/165368288-4709d491-66f6-4b04-81de-b5d0c530f421.png)


This repository includes:
* Source codes of BioGAN
* The trained weight for head translation of lab-taken images to the field taken images
* Jupyter notbooks for training and inference

# Getting Started
For getting started, install the required packages as listed below:

- tensorflow r1.0 or higher version
- numpy 1.11.0
- scipy 0.17.0
- pillow 3.3.0


## Reference
- The tensorflow implementation of CycleGAN, https://github.com/nanfengpo/CycleGAN-tensorflow-1?msclkid=d1fd27c8adbc11ec9a2f9153bce627cd
- The tensorflow implementation of pix2pix, https://github.com/yenchenlin/pix2pix-tensorflow
