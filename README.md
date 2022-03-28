# DM2022-LinearTransformers

## About this Project
This Repository contains a replication of the paper *[Transformers are RNNs:
Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/pdf/2006.16236v3.pdf)* from Angelos Katharopoulos, Apoorv Vyas , Nikolaos Pappas and François Fleuret. The paper and the associated Code on GitHub can be accessed on the following [website](https://linear-transformers.com).

In this Project we created a Linear Transformer in Julia in order to run a copy task and test the convergence of different sequence lengths. The basis of our code is the Julia package [Transformers.jl](https://github.com/chengchingwen/Transformers.jl). We changed the softmax-based attention function into a linear attention function and compared the original Transformer with the Linear Transformer on their convergence depending on sequence length. 

## Structure of the Repository
### Original Source Code in Python
The original Python code of the copy task done by a Linear Transformer can be found [here](https://github.com/idiap/linear-transformer-experiments/tree/master/causal-copy). We also included this example in our Templates folder under /Templates/Python Files/.
You need to install [torch](https://pytorch.org) and [pytorch-fast-transformers](https://github.com/idiap/fast-transformers) to run it. These packages can be downloaded by the following command:
```
pip install torch pytorch-fast-transformers
```
### Templates in Julia
Our main Template in Julia was the code of the copy task of the package Transformers.jl. All the relevant files to view the code of it and to understand our implementation better can be found at /Templates/Transformer/.
Furthermore we already did some work on Transformer.jl in our lecture Data Mining at Martin-Luther-University Halle- Wittenberg. The code of that can be accessed at /Templates/exercises of lecture/ as a jupyter notebook or a julia file.

### Replication of Linear Transformer in Julia
Our main file of the Linear Transformer in Julia is LinTrans.jl. We recommend opening this file in a Pluto notebook. This Code works on a CPU, but we strongly recommend using a GPU to lower computational time.
You can change the basic parameters, like sequence length, right on top of the code. After that make sure to run the whole code again. Otherwise Pluto could create an error.
The following packages are needed to run it: Flux, CUDA (if you use a GPU), Tullio, NNlib, Plots.
We recomment using the master branch. However, our test files and tryouts are accessable in the branches JB-working and vado.

### Article of our Replication
We created this replication as an examination in our lecture Data Mining. In the folder /Article you can find our text about this replication, were we further discuss our ideas for this replication.
