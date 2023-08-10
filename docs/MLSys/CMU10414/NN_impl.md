# Neural Network Implementation
## 1. Mutating the data field of a needle Tensor
直接按照下图这样写代码，会有什么问题吗？
![Alt text](assets/image-28.png)

会有，因为grad也是`Tensor`，所以每次做`w = w + (-lr) * grad`时，也够早了计算图。 

!!! note "1. Weight update doesn't need to be tracked"
    不需要把weight update也加到计算图中，因为weight update不需要梯度。

正确的做法是：只改值，不创造计算图：
![Alt text](assets/image-29.png)

## 2. Numerical Stability

浮点数的精度有限，浮点运算例如`0.4 - 0.1`得到的结果可能不是`0.3`，而是`0.2999993`。

以最常见的`softmax`操作为例，其中的`exp`如果输入的值很大，那么`exp`的结果就会是`nan`,所以需要一些技巧性的操作：

![Alt text](assets/image-30.png)

## 3. Design a Neural Network Library

用`Parameter`类来包装`Tensor`，这样可以把`Parameter`类的`grad`属性设置为`None`，这样就不会把`grad`也加到计算图中了。

