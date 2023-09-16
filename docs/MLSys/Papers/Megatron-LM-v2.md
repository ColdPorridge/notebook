[Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM(SC'21)](https://dl.acm.org/doi/10.1145/3458817.3476209)

![Alt text](assets/image-69.png)

## 1. Introduction

<figure markdown>
  ![Alt text](assets/image-57.png){ width="300" }
</figure>

Our approach allows us to perform training iterations on a model with 1 trillion parameters at 502 petaFLOP/s on 3072 GPUs (per-GPU throughput of 52% of theoretical peak).

![Alt text](assets/image-58.png)


## 2. Method

Notation:

- $(p, t, d)$ : Parallelization dimensions. $p$ for the pipeline-modelparallel size, $t$ for the tensor-model-parallel size, and $d$ for the data-parallel size.

- $n$ : Number of GPUs. We require $p \cdot t \cdot d=n$.

- B: Global batch size (provided as input).

- $b$ : Microbatch size.

- $m=\frac{1}{b} \cdot \frac{B}{d}:$ Number of microbatches in a batch per pipeline.


### 2.1 PTD-P
![Alt text](../assets/image-36.png)

### 2.2 interleaved pipelining schedule

!!! note "Pipeline Schedule Strategy"
    Assume 1 batch partitioned to $m$ mirco-batch, there are $p$ devices

    === "all-forward, all-backward"
        <figure markdown>
        ![Alt text](../assets/image-43.png)
        </figure>

        - bubble ratio(bubble/not-bubble): $\frac{p-1}{m}$
    
    === "1F1B"
        <figure markdown>
        ![Alt text](../assets/image-46.png)
        </figure>

        - bubble ratio(bubble/not-bubble): $\frac{p-1}{m}$

        - This schedule requires activations to be stashed for $p$ or fewer microbatches, rather than $m$ microbatches in all-forward, all-backward.
        
    === "1F1B with Interleaved Stages"
        <figure markdown>
        ![Alt text](assets/image-66.png)
        </figure>
        ??? example
            === "1F1B"
                ![Alt text](assets/image-67.png)
            === "1F1B with Interleaved Stages"
                v = 2

                ![Alt text](assets/image-68.png)

        - bubble ratio(bubble/not-bubble): $\frac{1}{v} \dot \frac{p-1}{m}$

        - the amount of communication also increases by $v$.

### 2.3 Scatter/gather communication optimization
![Alt text](assets/image-63.png)

## 3. Experiments and Analysis

- Run on [NVIDIA Selene Supercomputer](https://blogs.nvidia.com/blog/2020/12/18/nvidia-selene-busy/), Selene is composed of four SuperPODs, each with a total of 140 nodes, each a NVIDIA DGX A100, giving Selene a total of 560 nodes.

- In this work, use 384 nodes of Selene, each with 8 NVIDIA DGX A100 GPUs, for a total of 3072 GPUs.

??? question "intra-server communication"
    600GB/s of GPU-to-GPU bidirectional bandwidth?

- Each cluster node has 8 NVIDIA 80-GB A100 GPUs, connected to each other by NVLink and NVSwitch.

- Each node has eight NVIDIA Mellanox 200Gbps HDR Infiniband HCAs for application communication, with an additional two HCAs per node for dedicated storage.

- The nodes are connected in a threelevel (leaf, spine, core) fat-tree topology with 850 switches.

- The cluster uses an all-NVME shared parallel filesystem for high-performance data access and storage.

- When training a trillion-parameter model on 3072 GPUs, our implementation used an effective bisection bandwidth of 892 GB/s for pipeline-parallel communication, and 13 TB/s for data-parallel communication.

### 3.1 PTD-P Performance

![Alt text](assets/image-58.png)


### 3.2 Estimating end-to-end training time

- $P$: the number of parameters in the model

- $T$: the number of training tokens

- $X$: the number of FLOPs per GPU

- $n$: the number of GPUs

$$
\text { End-to-end training time } \approx \frac{8 T P}{n X}
$$

!!! example
    === "GPT-3"
        $P$ = 175 billion, $T$ = 300 billion, $n$ = 1024 A100 GPUs, achieve $X$ = 140 teraFLOP/s per GPU

        -> 34 days
    
    === "1 trillion parameter model"
        $P$ = 1 trillion, $T$ = 450 billion, $n$ = 3072 A100 GPUs, achieve $X$ = 163 teraFLOP/s per GPU

        -> 84 days
    
    > We believe these training times (using a reasonable number of GPUs) are practical.

### 3.3 Pipeline parallelism scalability

How well does pipeline parallelism scale for a **given batch size**?
<figure markdown>
  ![Alt text](assets/image-59.png){ width="500" }
</figure>

We evaluate the scaling of the default noninterleaved pipeline-parallel schedule using a weak scaling setup, a GPT model with 128 attention heads and a hidden size of 20480, and a **microbatch size of 1**.

We use a **tensor-parallel size of 8** for all configurations, and vary the total number of A100 GPUs used from 8 to 64.

with a pipelineparallel size of 1, we use a model with 3 transformer layers and 15 billion parameters, and with a pipeline-parallel size of 8, we use a model with 24 transformer layers and 121 billion parameters.

### 3.4 Impact of interleaved schedule and Scatter/gather communication optimization

!!! example ""
    === "impact of interleaved schedule"
        <figure markdown>
          ![Alt text](assets/image-60.png){ width="500" }
        </figure>
        
        This gap closes as the batch size increases due to two reasons: 
        
        - as the batch size increases, the bubble size in the default schedule decreases
    
        - the amount of point-to-point communication within the pipeline is proportional to the batch size, and consequently the non-interleaved schedule catches up as the amount of communication increases

    === "Scatter/gather communication optimization"
        <figure markdown>
          ![Alt text](assets/image-64.png){ width="500" }
        </figure>

### 3.5 How to Set parallelization dimensions? 

#### 3.4.1 Tensor versus Pipeline Parallelism

Evaluate the impact of pipeline and tensor model parallelism on performance for **a given model and batch size**.
<figure markdown>
  ![Alt text](assets/image-61.png){ width="500" }
</figure>
- We observe that tensor model parallelism is best within a node (DGX A100 server) due to its expensive all-reduce communication.

- Pipeline model parallelism, on the other hand, uses much cheaper point-to-point communication that can be performed across nodes without bottlenecking the entire computation.

!!! success "Takeaway #1"
    When considering different forms of model parallelism, tensor model parallelism should generally be used up to degree $g$ when using $g$-GPU servers, and then pipeline model parallelism can be used to scale up to larger models across servers.

#### 3.4.2 Pipeline versus Data Parallelism
For simplicity, we keep the microbatch size equal to 1 in these experiments.

<figure markdown>
  ![Alt text](assets/image-62.png){ width="500" }
</figure>

Pipeline bubble ratio: 

$$
\frac{p-1}{m}=\frac{n / d-1}{b^{\prime} / d}=\frac{n-d}{b^{\prime}} .
$$

Overall throughput will thus increase if the all-reduce communication needed for data parallelism does not drastically increase with higher $d$, which should hold since the communication time for a ring-based implementation scales with $\frac{d-1}{d}$.

!!! success "Takeaway #2"
    When using data and model parallelism, a total model-parallel size of $M = t \dot p$ should be used so that the model’s parameters and intermediate metadata fit in GPU memory; data parallelism can be used to scale up training to more GPUs.

### 3.6 How to set microbatch size?
<figure markdown>
  ![Alt text](assets/image-65.png){ width="500" }
</figure>

Increasing the microbatch size decreases the number of microbatches in the pipeline ($m$):

- leading to a larger pipeline bubble;

- improve GPU utilization by increasing the arithmetic intensity of executed kernels.

Two factors are at odds with each other, which makes the choice of optimal microbatch size challenging.

??? note "Theoretical analysis"
    The total time spent computing a batch, ignoring communication cost, is
    $$
    \left(B /(d \dot b) +p-1\right) \cdot\left(t_f(b)+t_b(b)\right)
    $$

    Need choose a proper $b$ to minimize the above equation.

!!! success "Takeaway #3"
    The optimal microbatch size $b$ depends on the throughput and memory footprint characteristics of the model, as well as the pipeline depth $p$, data-parallel size $d$, and batch size 𝐵.
