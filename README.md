# DM2022-LinearTransformers

## About this Project
This Repository contains a replication of the paper *[Transformers are RNNs:
Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/pdf/2006.16236v3.pdf)* from Angelos Katharopoulos, Apoorv Vyas , Nikolaos Pappas and François Fleuret. The paper and the associated Code on GitHub can be accessed on the following [website](https://linear-transformers.com).

In this Project we created a Linear Transformer in Julia in order to run a copy task and test the convergence of different sequence lengths. The basis of our code is the Julia package [Transformers.jl](https://github.com/chengchingwen/Transformers.jl). We changed the softmax-based attention function into a linear attention function and compared the original Transformer with the Linear Transformer on their convergence depending on sequence length. 

## Structure of the Repository
### Original Source Code in Python
The original Python code of the copy task done by a Linear Transformer can be found [here](https://github.com/idiap/linear-transformer-experiments/tree/master/causal-copy). 
You need to install torch and pytorch-fast-transformers to run it. These packages can be downloaded by the following command:
```
pip install torch pytorch-fast-transformers
```

### Replication of Linear Transformer in Julia
Our main file of the Linear Transformer in Julia is: …. 
You can change the basic parameters by…. 
The following packages are needed to run it:
We recomment using the master branch. However, our test files and tryouts are accessable in the branches JB-working and vado.
