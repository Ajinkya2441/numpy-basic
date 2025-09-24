import numpy as np

# ==========================================================
# 1. ARRAY CREATION
# ==========================================================
print("ARRAY CREATION")
print(np.array([1,2,3]))                 # from list
print(np.asarray([4,5,6]))               # asarray
print(np.arange(0,10,2))                 # range
print(np.linspace(0,1,5))                # evenly spaced
print(np.logspace(1,3,3))                # log spaced
print(np.geomspace(1,1000,4))            # geometric progression
print(np.zeros((2,3)))                   # zeros
print(np.ones((2,3)))                    # ones
print(np.empty((2,3)))                   # empty
print(np.full((2,3),7))                  # full
print(np.identity(3))                    # identity
print(np.eye(3,4,k=1))                   # eye with shift
print(np.diag([1,2,3]))                  # diagonal
print(np.tri(3))                         # lower triangle
print(np.tril([[1,2],[3,4]]))            # lower part
print(np.triu([[1,2],[3,4]]))            # upper part
print(np.meshgrid([1,2],[3,4]))          # meshgrid
print(np.fromfunction(lambda i,j: i+j,(2,3),dtype=int))
print(np.fromiter(range(5),dtype=int))
print(np.indices((2,3)))

# Random arrays
print(np.random.rand(2,3))
print(np.random.randn(2,3))
print(np.random.randint(1,10,(2,3)))
print(np.random.random_sample((2,2)))
print(np.random.choice([1,2,3],5))
arr = np.arange(5)
np.random.shuffle(arr)
print(arr)
print(np.random.permutation([1,2,3,4]))

# ==========================================================
# 2. ARRAY ATTRIBUTES & INSPECTION
# ==========================================================
print("\nARRAY ATTRIBUTES")
a = np.array([[1,2,3],[4,5,6]])
print(a.shape, a.ndim, a.size, a.dtype, a.itemsize, a.nbytes)
print(np.isscalar(5), np.iscomplex([1+2j,3]))
print(np.isreal([1+2j,3]))
print(np.isfinite([1,np.inf,np.nan]))

# ==========================================================
# 3. INDEXING & SLICING
# ==========================================================
print("\nINDEXING & SLICING")
b = np.arange(10)
print(b[2])
print(b[2:7])
print(b[::-1])
print(b[[1,3,5]])
print(b[b%2==0])
print(np.take(b,[1,4,7]))
c = np.array([10,20,30,40])
np.put(c,[0,2],[99,77])
print(c)
print(np.choose([0,1,0],[ [1,2,3],[4,5,6] ]))
print(np.where(b>5))
print(np.nonzero(b))
print(np.argwhere(b>5))
print(np.compress([0,1,0,1],c))

# ==========================================================
# 4. ARRAY MANIPULATION
# ==========================================================
print("\nARRAY MANIPULATION")
d = np.arange(6)
print(d.reshape(2,3))
print(d.ravel())
print(d.flatten())
print(np.squeeze([[1],[2],[3]]))
print(np.transpose([[1,2,3],[4,5,6]]))
print(np.swapaxes([[1,2],[3,4]],0,1))
print(np.moveaxis(np.ones((1,2,3)),0,-1).shape)
print(np.concatenate([d,d]))
print(np.stack([d,d]))
print(np.hstack([d,d]))
print(np.vstack([d,d]))
print(np.dstack([d,d]))
print(np.column_stack(([1,2,3],[4,5,6])))
print(np.row_stack(([1,2,3],[4,5,6])))
print(np.split(d,3))
print(np.hsplit(np.arange(12).reshape(2,6),3))
print(np.vsplit(np.arange(12).reshape(6,2),3))
print(np.tile([1,2],3))
print(np.repeat([1,2],3))

# ==========================================================
# 5. MATHEMATICAL OPERATIONS
# ==========================================================
print("\nMATH OPERATIONS")
x = np.array([1,2,3])
y = np.array([4,5,6])
print(x+y, x-y, x*y, x/y)
print(np.add(x,y), np.subtract(x,y))
print(np.multiply(x,y), np.divide(y,x))
print(np.mod(y,x))
print(np.power(x,2))
print(np.reciprocal([1,2,3]))
print(np.sin(x), np.cos(x), np.tan(x))
print(np.arcsin([0,1]))
print(np.sinh([0,1]))
print(np.floor([1.7,-1.7]))
print(np.ceil([1.2,-1.2]))
print(np.trunc([1.5,-1.5]))
print(np.rint([1.5,2.5]))
print(np.fix([1.5,-1.5]))
print(np.exp([1,2]))
print(np.expm1([1,2]))
print(np.log([1,np.e,np.e**2]))
print(np.log10([1,10,100]))
print(np.log2([1,2,4]))
print(np.log1p([0,1]))
print(np.sqrt([1,4,9]))
print(np.cbrt([1,8,27]))
print(np.square([2,3]))
print(np.abs([-1,2,-3]))
print(np.sign([-10,0,10]))
print(np.clip([-5,0,5],-1,1))

# ==========================================================
# 6. AGGREGATE / STATISTICS
# ==========================================================
print("\nAGGREGATES")
m = np.array([[1,2,3],[4,5,6]])
print(m.sum(), m.prod())
print(m.mean(), m.std(), m.var())
print(m.min(), m.max(), np.ptp(m))
print(m.argmin(), m.argmax())
print(m.cumsum(), m.cumprod())
print(np.median(m), np.percentile(m,50))
print(np.quantile(m,0.25))

# ==========================================================
# 7. LINEAR ALGEBRA
# ==========================================================
print("\nLINEAR ALGEBRA")
p = np.array([[1,2],[3,4]])
q = np.array([[5,6],[7,8]])
print(np.dot(p,q))
print(np.matmul(p,q))
print(np.vdot([1,2,3],[4,5,6]))
print(np.inner([1,2],[3,4]))
print(np.outer([1,2],[3,4]))
print(np.cross([1,2,3],[4,5,6]))
print(np.tensordot([1,2],[3,4],axes=1))
print(np.linalg.det(p))
print(np.linalg.inv(p))
print(np.linalg.eig(p))
print(np.linalg.eigvals(p))
print(np.linalg.svd(p))
print(np.linalg.qr(p))
print(np.linalg.cholesky([[1,0],[0,1]]))
print(np.linalg.solve(p,[5,6]))
print(np.linalg.lstsq(p,[5,6],rcond=None))
print(np.linalg.matrix_power(p,2))
print(np.linalg.matrix_rank(p))
print(np.linalg.norm([3,4]))
print(np.linalg.cond(p))

# ==========================================================
# 8. RANDOM (more distributions)
# ==========================================================
print("\nRANDOM DISTRIBUTIONS")
print(np.random.binomial(10,0.5,5))
print(np.random.poisson(5,5))
print(np.random.normal(0,1,5))
print(np.random.uniform(0,1,5))
print(np.random.beta(2,5,5))
print(np.random.gamma(2,2,5))
print(np.random.chisquare(2,5))
print(np.random.exponential(1,5))
print(np.random.laplace(0,1,5))
print(np.random.logistic(0,1,5))
print(np.random.multinomial(10,[0.2,0.8]))
print(np.random.multivariate_normal([0,0],[[1,0],[0,1]],2))
print(np.random.pareto(1,5))
print(np.random.zipf(2,5))

# ==========================================================
# 9. SORTING & SET OPS
# ==========================================================
print("\nSORTING & SET OPS")
s = np.array([5,2,9,1])
print(np.sort(s))
print(np.argsort(s))
print(np.lexsort((s,)))
print(np.partition(s,2))
print(np.argpartition(s,2))
print(np.searchsorted([1,2,3,4,5],3))
print(np.unique([1,2,2,3]))
print(np.intersect1d([1,2,3],[2,3,4]))
print(np.union1d([1,2],[2,3]))
print(np.setdiff1d([1,2,3],[3,4]))
print(np.setxor1d([1,2,3],[2,3,4]))
print(np.in1d([1,2,3],[2,4]))

# ==========================================================
# 10. FILE I/O
# ==========================================================
print("\nFILE I/O")
np.savetxt("data.txt", np.array([1,2,3]))
print(np.loadtxt("data.txt"))

# ==========================================================
# 11. SPECIAL FUNCTIONS
# ==========================================================
print("\nSPECIAL FUNCTIONS")
print(np.meshgrid([1,2],[3,4]))
print(np.ogrid[0:3,0:3])
print(np.mgrid[0:3,0:3])
print(np.diagflat([1,2,3]))
print(np.histogram([1,2,1,2,3],bins=3))
print(np.histogram2d([1,2,1],[1,2,2],bins=2))
print(np.digitize([0.2,6.4,3.0,1.6],bins=[0,1,3,5,7]))
print(np.bincount([0,1,1,2,2,2]))
