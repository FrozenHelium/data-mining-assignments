import math
import numpy as np

class SVD:
    def proj(self, u, v):
        return u * np.dot(v,u) / np.dot(u,u)

    # orthonormalize the given matrix using  Gram-Schmidt orthonormalization algorithm
    def gs_orthonormalize(self, v):
        v = 1.0 * v
        u = np.copy(v)
        for i in xrange(1, v.shape[1]):
            for j in xrange(i):
                u[:,i] -= self.proj(u[:,j], v[:,i])
        den = (u**2).sum(axis=0) **0.5
        return u/den


    # decomposes matrix A into USV'
    # U,V are orthogonal matrices, S is a diagonal matrix
    def decompose(self, a):
        # In order to find U, we have to start with AA'
        a_at = a*a.T
        # eigen value & vector of A
        ei__a_at, ev__a_at = np.linalg.eig(a_at);
        temp = []
        sort_indices = ei__a_at.argsort()[::-1]
        ev__a_at = ev__a_at.transpose()
        for n, i in enumerate(sort_indices):
            temp.append(np.array(ev__a_at[i])[0].tolist())
        u = np.matrix(temp).transpose()

        # similarly for V
        at_a = a.T*a
        ei__at_a, ev__at_a = np.linalg.eig(at_a);
        temp = []
        sort_indices = ei__at_a.argsort()[::-1]
        ev__at_a = ev__at_a.transpose()
        for n, i in enumerate(sort_indices):
            temp.append(np.array(ev__at_a[i])[0].tolist())
        v = np.matrix(temp).transpose()

        # now for S, we fill diagonal by square root of eigen values
        # The non-zero eigenvalues of U and V are always the same,
        # so, it doesn't matter which one we take them from
        s = []
        for i, eu in enumerate(ei__a_at):
            temp = []
            for j, ev in enumerate(ei__at_a):
                if( i == j):
                    temp.append(np.sqrt(eu))
                else:
                    temp.append(0)
            s.append(temp)
        s = np.matrix(s)
        return u, s, v

    def reduce(self, a, dim):
        u, s, v = self.decompose(a)
        s = s[:dim, :dim]
        u = u[:, :dim]
        v = v[:dim, :]
        temp = u*s*v
        return temp

svd = SVD()
A = np.matrix([ [2, 0, 8, 6, 0], [1, 6, 0, 1, 7], [5, 0, 7, 4, 0], [7, 0, 8, 5, 0], [0, 10, 0, 0, 7] ])
print(A)
print(' ')
A = svd.reduce(A, 3)
print(A)
