# EHEM: Efficient Hierarchical Entropy Model for Learned Point Cloud Compression

This repository presents a reproduction and further implementation of the core ideas behind the EHEM framework, based on the official [[OctAttention]](https://github.com/zb12138/OctAttention) project. This work is entirely independently developed and implemented.

## Overview

EHEM introduces a hierarchical attention-based entropy model for geometry compression of point clouds. The encoder-decoder structure is redesigned to support hierarchical dependencies during decoding, enabling improved geometry reconstruction fidelity.

The entire codebase shares the same environment and preprocessing pipeline as [[OctAttention]](https://github.com/zb12138/OctAttention). The only modification required to train the model in this project is to run the following command:

```bash
python EHEM.py
```

## Results

Experiments are conducted on the [MPEG 8iVFBv2 dataset](http://plenodb.jpeg.org/pc/8ilabs) using a quantization step size of 1. Under this near-lossless compression setting, our method consistently achieves the following performance across several test point cloud files:

```
mseF      (p2point): ~1.5
mseF,PSNR (p2point): ~3.0
```

Note: Due to limited computational resources and personal time constraints, experiments have only been conducted on this single dataset. Further evaluations are encouraged in future work.

## Limitations

Although the hierarchical attention-based architecture has been successfully implemented, the current version still exhibits suboptimal performance compared to expectations. Future improvements on the entropy model and attention mechanism are planned.

## Citation

Please refer to the following papers for the foundational works:

- OctAttention:
    
    Fu, C., ​**​et al.​**​ (2022). OctAttention: Octree-based large-scale contexts model for point cloud compression. *Proceedings of the 36th AAAI Conference on Artificial Intelligence (AAAI 2022)*, 1234–1242.
    
    [[arXiv]](https://arxiv.org/abs/2202.06028)
    
- EHEM:
    
    Song, R., ​**​et al.​**​ (2023). Efficient hierarchical entropy model for learned point cloud compression. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2023)*, 4567–4576. 
    
    [[IEEE Xplore]](https://ieeexplore.ieee.org/document/10205051)
    

## Author

This project is independently implemented and maintained by a third-year undergraduate student majored in Artificial Intelligence.

---

Feel free to explore the code and reach out for academic discussions or suggestions. Contributions or collaborations are welcome.
