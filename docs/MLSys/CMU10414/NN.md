# Neural Network Abstraction
省流图：
![Alt text](assets/image-12.png)
## 1. Programming abstractions
以Caffe，TensorFlow，PyTorch为例，介绍神经网络的编程抽象：
![Alt text](assets/image.png)
### 1.1 Caffe
**单纯的前向传播和反向传播**
![Alt text](assets/image-1.png)

### 1.2 TensorFlow
![Alt text](assets/image-2.png)

### 1.3 PyTorch
![Alt text](assets/image-3.png)

## 2. High level modular library components
### 2.1 三大组件
**主要要实现三大类组件：**
![Alt text](assets/image-4.png)

**深度学习本身就是很模块化的，所以用模块化的方式来构建深度学习库是很自然的事情。**

![Alt text](assets/image-5.png)

**使用nn.Module作为基础。**
![Alt text](assets/image-6.png)

**损失函数也可以被视为一个nn.Module。**
![Alt text](assets/image-7.png)

**优化器：更新权重，以及维护用于更新权重的其他一些“状态”（不同的优化算法的怎神的要求）。**
![Alt text](assets/image-8.png)

??? Note "优化器喝喝实现REgularization"
    ![Alt text](assets/image-9.png)

### 2.2 Initialization
既不能太小——可能梯度消失；也不能太大——可能梯度爆炸。

![Alt text](assets/image-10.png)


### 2.3 Data loader and preprocessing
**Tianqi在这里也只是简单提了一下，感觉没说啥**
![Alt text](assets/image-11.png)

## 3. Summary
能看到现在的pytorch相较于caffe的优越性：

- Caffe 使用的Layer，既是梯度计算的基本组件，也是模型组合的基本组件，没有把这两者解耦开来(我的理解是：我自己写个Module还得负责写它的bakcward？)
  ![Alt text](assets/image-13.png)

- Pytorch 使用的Module，把梯度计算和模型组合解耦开来了，这样就可以更加灵活的组合模型了。
  ![Alt text](assets/image-14.png)