## 1. NDArray
可以通过`device`参数来指定NDArray的存储位置，如果不指定，默认是在CPU上的。

```python
from needle import backend_ndarray as nd
x = nd.NDArray([1, 2, 3], device = nd.cuda)
y = x + 1
```
首先看一下`NDArray`的结构, An NDArray contains the following fields:
  
- handle: The backend handle that build a flat array which stores the data.
    
- shape: The shape of the NDArray
    
- strides: The strides that shows how do we access multi-dimensional elements
    
- offset: The offset of the first element.
    
- device: The backend device that backs the computation

`y`也会是在GPU上的NDArray，`x + 1`的过程是在GPU上完成的。

看一下这个过程中都发生了什么，对调用链进行梳理：
!!! note "调用链梳理"
    === "frame 1"
        从NDArray的构造函数开始：此处是先进入第三个分支，再进入第二个分支调用`self.make`、`array.device.from_numpy`，最终调用了`self._init`

        可以看到，一个新的NDArray最主要的backend的实体的创建，是通过`array.device.Array(prod(shape))`，这里根据device就会调用不同的backend的实现、
        ```python
        import numpy as np

        class NDArray:
            def __init__(self, other, device=None):
                """ Create by copying another NDArray, or from numpy """
                if isinstance(other, NDArray):
                    # create a copy of existing NDArray
                    if device is None:
                        device = other.device
                    self._init(other.to(device) + 0.0)  # this creates a copy
                elif isinstance(other, np.ndarray):
                    # create copy from numpy array
                    device = device if device is not None else default_device()
                    array = self.make(other.shape, device=device)
                    array.device.from_numpy(np.ascontiguousarray(other), array._handle)
                    self._init(array)
                else:
                    # see if we can create a numpy array from input
                    array = NDArray(np.array(other), device=device)
                    self._init(array)
        
            def _init(self, other):
                self._shape = other._shape
                self._strides = other._strides
                self._offset = other._offset
                self._device = other._device
                self._handle = other._handle
            
            @staticmethod
            def make(shape, strides=None, device=None, handle=None, offset=0):
                """ Create a new NDArray with the given properties.  This will allocation the
                memory if handle=None, otherwise it will use the handle of an existing
                array. """
                array = NDArray.__new__(NDArray)
                array._shape = tuple(shape)
                array._strides = NDArray.compact_strides(shape) if strides is None else strides
                array._offset = offset
                array._device = device if device is not None else default_device()
                if handle is None:
                    array._handle = array.device.Array(prod(shape))
                else:
                    array._handle = handle
                return array
        ```

    === "frame 2"
        施工中

## 2. Transformation as Strided Computation

**reshape操作：**
```python
import numpy as np
x = nd.NDArray([0, 1,2,3,4,5], device=nd.cpu_numpy())

y = nd.NDArray.make(shape=(2, 3), 
            strides=(2, 1), 
            device=x.device, 
            handle=x._handle, 
            offset=0)

```

```
y: NDArray([[0, 1, 2], [3, 4, 5]], device = cpu_numpy())
```


从这个例子也可以看到`NDArray.make`的作用，因为`handle = x._handle`，所以这里其实是对`x`进行了一次浅拷贝，`y`和`x`共享同一个`handle`（同一块backend的内存），所以`y`的改变会影响到`x`。`y`是一个strided view of `x`。

也可以只选取`x`的一部分来构造`z`：

**slice操作：**
```python
z = nd.NDArray.make(shape=(2, 2), 
            strides=(3, 1), 
            device=x.device, 
            handle=x._handle, 
            offset=1)
```

```
z: NDArray([[1, 2], [4, 5]], device = cpu_numpy())
```

**transpose操作：**
```python
b = nd.NDArray.make(shape=(3, 2), 
            strides=(1, 3), 
            device=y.device, 
            handle=y._handle, 
            offset=0)
```

```
b: NDArray([[0, 3], [1, 4], [2, 5]], device = cpu_numpy())
```

**broadcast操作：**
```python
c = nd.NDArray.make(shape=(2, 3, 4), 
            strides=(3, 1, 0), 
            device=y.device, 
            handle=y._handle, 
            offset=0)
```

```
c: NDArray([[[0, 0, 0, 0],
             [1, 1, 1, 1],
             [2, 2, 2, 2]],
            [[3, 3, 3, 3],
             [4, 4, 4, 4],
             [5, 5, 5, 5]]], device = cpu_numpy())
```

!!! note "compact"
    === "为什么需要compact"
        注意，虽然这些方法可以得到各样的view，但是在做计算的时候，还是直接会按照底层的存储形式做计算的，例如以CPU作为backend的时候，`EwiseAdd`的实现方式是：
        ```cpp
        void EwiseAdd(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
            /**
            * Set entries in out to be the sum of correspondings entires in a and b.
            */
            for (size_t i = 0; i < a.size; i++) {
                out->ptr[i] = a.ptr[i] + b.ptr[i];
            }
        }
        ```
        这可能导致问题，例如我在做`z = z + 1`的时候，我希望只是`z`中的4个位置的值加1，但是实际上会把6个位置的值都加1（因为`z`是`x`的一个view，`x`的`handle`中存储的是6个值）

    === "什么是compact"
        `NDArray`类中有两个方法`is_compact`和`compact`，前者用来判断一个`NDArray`是否是compact的，后者用来把一个`NDArray`转换成compact的。
        
        ```python
        def is_compact(self):
        """ Return true if array is compact in memory and internal size equals product
        of the shape dimensions """
            return (self._strides == self.compact_strides(self._shape) and
                prod(self.shape) == self._handle.size)

        ```
    === "如何变成compact"
        make一个新的，此时做深拷贝将`view`给`materialize`（Lazy策略）
        ```python
        def compact(self):
            """ Convert a matrix to be compact """
            if self.is_compact():
                return self
            else:
                out = NDArray.make(self.shape, device=self.device)
                self.device.compact(
                    self._handle, out._handle, self.shape, self.strides, self._offset
                )
                return out
        ```

## 3. CUDA Acceleration

通过在`ndarray_vackend_cuda.cu`中实现一些算子，来进行加速。

这里以`Add`为例，看一下实现的具体流程，有如下代码
```cuda
__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);

  m.def("ewise_add", EwiseAdd); // 把EwiseAdd暴露给python
  m.def("scalar_add", ScalarAdd); // 把ScalarAdd暴露给python
}
```

这样这些就可以作为python的函数来调用了，在`ndarray.py`中，有如下代码：
```python
def __add__(self, other):
    return self.ewise_or_scalar(
        other, self.device.ewise_add, self.device.scalar_add
    )
```
重载了`+`操作符，这样就可以直接用`+`来做加法了，如果`device`是`cpu_numpy`，那么就会调用`cpu_numpy`的实现，如果是`cuda`，就会调用`cuda`的实现。



