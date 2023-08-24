[Mixed Precision Training](https://arxiv.org/pdf/1710.03740.pdf)

https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html


增加神经网络的大小通常会提高准确性，但也会增加训练模型的内存和计算要求

我们介绍了使用半精度浮点数（half-precision floating point numbers）训练深度神经网络的方法，而不会损失模型精度且不必修改超参数。



权重、激活和梯度以 IEEE 半精度格式存储。由于这种格式的范围比单精度更窄，我们提出了三种技术来防止关键信息丢失。