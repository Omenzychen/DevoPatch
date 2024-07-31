# [[IEEE TIFS] Query-Efficient Decision-Based Black-Box Patch Attack](https://ieeexplore.ieee.org/abstract/document/10227335)

Zhaoyu Chen, Bo Li, Shuang Wu, Shouhong Ding, Wenqiang Zhang

This repository offers Pytorch code to reproduce results from the paper. Please consider citing our paper if you find it interesting or helpful to your research.

```
@ARTICLE{devopatch2023chen,
  author={Chen, Zhaoyu and Li, Bo and Wu, Shuang and Ding, Shouhong and Zhang, Wenqiang},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Query-Efficient Decision-Based Black-Box Patch Attack}, 
  year={2023},
  volume={18},
  number={},
  pages={5522-5536},
  keywords={Closed box;Perturbation methods;Glass box;Evolutionary computation;Robustness;Image classification;Optimization;Adversarial example;patch attack;black-box attack;differential evolutionary algorithm},
  doi={10.1109/TIFS.2023.3307908}}
```


## Quick Start

- **Attack**

  Attack demo for one image.

   ```bash
  CUDA_VISIBLE_DEVICES=0 python3 devopatch_attack.py --model resnet50 --bs 1000 --pop_size 10 --steps 10000 --init_rate 0.35 --mutation_rate 1
  ```


## License
The project is only free for academic research purposes but has no authorization for commerce. Part of the code is modified from Prompt-to-Prompt.