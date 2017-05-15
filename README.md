# HexSimulation

## File Structure

- `matlab/` directory containing original matlab code
- 

## Original Matlab Code (`HexClusterMF.m`)

## NumPy Tutorial
### [Basics](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)

- `ndarray.ndim` (number of axises)
- `ndarray.shape` (a tuple containing size of the array in each dimension)
- `ndarray.dtype` (type of elements in the array)

#### Array Creation

- from `list` or `tuple`
    - nested `list` or `tuple` will be transformed to array with higher dimension
    - `dtype` can also be specified when create an array
- `np.zeros((m, n))` and `np.ones((m, n))` create `m` by `n` matrix with `0s` or `1s`
- `np.arange(start, stop, step)` create an 1-D array with a range of elements
    - due to the finite precision of floating numbers, it is impossible to predict number of elements obtained.
- `np.linespace(start, stop, num)` create an 1-D array with `num` of elements 
- `np.fromfunction(f, dim, dtype)`
- `np.fromfile`

#### Operation

- Only `upcasting` is allowed
- `+, -, *, /` elementwise arithmetic operation 
    - `np.sin`, `np.cos`, `np.exp` are functions (universal functions) applied elementwise
    - `np.transpose`, `np.trace` are also provided
        - `A.T` is the same as `A.transpose`
    - `np.linalg.svd`
    
#### Indexing, Slicing and Iterating

- One index or indices (i.e. `begin:stop:step`) per axis and separated by `,` in the square bracket
    - For example, `b[1:3, : ]`
    - `...` represent as many colons as needed to produce a complete indexing tuple
- Iterating over multidimensional arrays is done with respect to the first axis
    - e.g. `for row in A`
    - To apply elements width operation we can use `np.array.flat`
        - e.g. `for e in A.flat`
        
#### Stacking together Different Array
omitted

#### Splitting Array

- `np.hsplit(arr, )` splits an array into many smaller arrays
    - `np.hsplit(arr, n)` splits into `n` arrays
    - `np.hsplit(arr, (x1, x2 .. xn))` splits alone `x1, x2, .. xn` col
    - `np.vsplit` can be applied similarly
    
#### View and Copy

- Slicing an array create a view of it (e.g. `a.view()`)  which looks the same data but doesn't own it
- `a.copy()` creates an deep copy of the data

#### Linear Algebra (`np.linalg`)

- `inv(A)` creates inverse of `A`
- `solve(A,b)` solves `Ax=b`
- `eig(A)` returns eigen values and corresponding normalized eigen vectos
- `eye(n)` creates an `n` by `n` identity matrix
- `svd` returns `U, S, V`. Notice that `S` is an 1-D array

### Reference
- [NumPy Reference]