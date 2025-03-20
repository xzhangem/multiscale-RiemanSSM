# Implementation of "Landmark-Free Anatomical Shape Analysis Using Explicit Latent Codes Learned from Deformation-Specific Shape Metric and Multiscale Training"  

Motivation: The project is motivated by application the theory of Sobolev shape metric (IJCV 2023, https://github.com/emmanuel-hartman/H2_SurfaceMatch) and its T-PCA version Bera-esa (IJCV 2024, https://github.com/emmanuel-hartman/BaRe-ESA) to medical shape data. However, we find that directly usage of this fancy theory to some high-deformed dataset where each sample may vary a lot from the others, such as pancreas dataset, will not obtain satisfying results. Therefore, we advance the original Bera-esa with multiscale training stragety, and its performance becomes great and can be tested better than sevaral existing results. 

The workflow of the proposed work is summarzied in the figure below:
![image](https://github.com/xzhangem/multiscale-RiemanSSM/blob/main/Figures/diagram_tpca.png)
For the part **A** of multiscale T-PCA training: 

`python mean_tpca.py --data_file <your dataset filename> --template_save <mean shape save name> --resolution <scale_num> --components_num <T-PCA mode num> --pca_save_name <T-PCA save name (in .npy format)>`

***NOTICE***: Pre-alignment using efficient method like ICP or CPD as preprocessing for raw dataset is suggested to get rid of the impact of basic transformation including transition, rotation and scaling for SSM, and you can active `pre_align` and specify the pre-align save file via `--prealign_file`. 

For the part **B** of the explicit latent code learning of shape, we use the trained pancreas T-PCA eigenvalues stored in TPCA-evec file as example:

`python ms_latent_code.py --TPCA_eigvec_prefix ./TPCA_evec/pancreas_tpca/pancreas_evec_1 --resolution 3 --input <input mesh file> --recon_dir <reconstructed mesh savename> --lc_dir <latent code savename (in .npy format)>`

For the part **C** of interpolation and and explolation with the latent codes, please refer to basis_interpolate(expolate).py file. 


