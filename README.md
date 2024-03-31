# Reimplementing-Context-Encoders-for-Image-Inpainting-Task
## Introduction
Image inpainting is filling in missing regions of an image using information from surrounding areas. The goal is to recover the image as accurately
as possible based on available information. In recent years, deep learning-based
methods have shown promising results in this task. The paper "Context Encoders:
Feature Learning by Inpainting" [1] proposed an encoder-decoder architecture for
the image inpainting task that can predict missing regions of an image. The proposed architecture has shown good performance on the StreetView and ImageNet
datasets. In this proposal, we aim to reimplement the same architecture on the
CelebA dataset to evaluate its performance on a different set of images. 

<figure>
  <img
  src="https://github.com/LiangSylar/Reimplementing-Context-Encoders-for-Image-Inpainting-Task/assets/64362092/feaa1531-a938-4b5f-b432-2d9bf4bf88a3"
  alt="Image."
  height="350">
  <figcaption>The architecture of the context encoders.</figcaption>
</figure> 
<be>


## References 
[1] Deepak Pathak et al. “Context Encoders: Feature Learning by Inpainting”. In: Proceedings of
the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). June 2016.
