# Implementation of "Landmark-Free Anatomical Shape Analysis Using Explicit Latent Codes Learned from Deformation-Specific Shape Metric and Multiscale Training"  

Motivation: The project is motivated by application the theory of Sobolev shape metric (IJCV 2023, https://github.com/emmanuel-hartman/H2_SurfaceMatch) and its T-PCA version Bera-esa (IJCV 2024, https://github.com/emmanuel-hartman/BaRe-ESA) to medical shape data. However, we find that directly usage of this fancy theory to some high-deformed dataset where each sample may vary a lot from the others, such as pancreas dataset, will not obtain satisfying results. Therefore, we advance the original Bera-esa with multiscale training stragety, and its performance becomes great and can be tested better than sevaral existing results. 

The workflow of the proposed work is summarzied in the figure below:
![image](https://github.com/xzhangem/multiscale-RiemanSSM/blob/main/Figures/diagram_tpca.png)
For the part `rgb(0,255,0)`A of multiscale T-PCA training: 

