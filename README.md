CNN Exploration and Applications

This repository houses a series of Jupyter notebooks and a folder that collectively explore various aspects of convolutional neural networks (CNNs) through both theoretical insights and practical implementations. The project covers topics ranging from image reconstruction, denoising, and segmentation to efficient sparse CNN designs and full-scale implementations in PyTorch and from scratch.

autoencoders_image_processing.ipynb
This notebook focuses on image reconstruction and denoising using popular datasets and autoencoder models. It demonstrates image reconstruction on the CIFAR-10 dataset by applying both a linear autoencoder and a convolutional autoencoder. For the MNIST dataset, it explores image denoising with similar autoencoder architectures. In addition, the notebook builds two CNN classifiers: one trained on original CIFAR-10 images and evaluated on data produced by the reconstruction models, and another doing the same for MNIST images with data from the denoising models. Finally, it investigates latent vector interpolation to reveal insights into the underlying feature representations.

image_segmentation.ipynb
This notebook delves into the realm of image segmentation using autoencoders. Using brain tumor images along with their corresponding ground truth masks, it studies segmentation techniques and demonstrates how autoencoders can be adapted to distinguish tumor regions from healthy tissue.

sparse_cnn.ipynb
Focusing on model efficiency, this notebook explores the concept of sparse convolutional neural networks. It implements techniques for sparsifying CNN layers to reduce computational overhead and model complexity, making it a valuable resource for research into lightweight and efficient network designs.

cnn_pytorch.ipynb
This hands-on guide provides a step-by-step approach to implementing CNNs using the PyTorch framework. It covers model building, training, and evaluation, offering practical insights into deep learning with one of the most widely-used libraries in the field.

From-Scratch CNN Implementation
In addition to the notebooks, the repository includes a dedicated folder containing a full, from-scratch implementation of a CNN. This section is ideal for users interested in understanding the inner workings of CNN architectures without relying on high-level frameworks, thereby offering a deeper dive into the algorithms and mathematical principles underlying CNNs.

Feel free to explore each section to deepen your understanding of CNNs and their applications in image processing tasks.
