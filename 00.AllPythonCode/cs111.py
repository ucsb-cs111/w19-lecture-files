# Python routines defined in CS 111 lecture, Winter 2019

########################################################################
# These are the standard imports for CS 111. 
# This list may change as the quarter goes on.
import os
import math
import numpy as np
import numpy.linalg as npla
import scipy
import scipy.sparse.linalg as spla
from scipy import sparse
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
# %matplotlib tk

########################################################################
def draw_fern(nsteps = 1000, speed = 100):
    """Draw the Barnsley fractal fern
    Parameters: 
      nsteps (default 1000): number of fern points to draw.
      speed (default 100): drawing speed in points per second. (not implemented yet)
    Outputs:
      none
    """

    # There are four affine transformations that go from one fern point to the next.
    # Each transformation is represented by a 2-by-2 matrix A and a 2-vector b.
    Alist = [ np.array( [[ .85,  .04], [-.04, .85]] ), \
              np.array( [[ .20, -.26], [ .23, .22]] ), \
              np.array( [[-.15,  .28], [ .26, .24]] ), \
              np.array( [[   0,    0], [   0, .16]] )]

    blist = [ np.array( [0, 1.6] ), \
              np.array( [0, 1.6] ), \
              np.array( [0, .44] ), \
              np.array( [0,   0] )]

    # Here are the probabilities that each transformation is chosen
    probs = [ .85, .07, .07, .01 ]

    # Start the list of points
    x = np.array([.5, .5])
    xcoords = [x[0]]
    ycoords = [x[1]]
    
    # Start the figure with the first point
    # for now, we just draw the figure at the end   

    # Add the rest of the points
    for step in range(nsteps):

        # Pause between points
        # for now, no pause

        # Choose a transformation for this step
        which_trans = np.random.choice(range(4), p = probs)
        A = Alist[which_trans]
        b = blist[which_trans]

        # Step to the next point
        x = A @ x + b

        # Update the list of points and the plot
        xcoords.append(x[0])
        ycoords.append(x[1])
        
    # Plot the final figure
    plt.plot(xcoords, ycoords, linestyle='', marker='.', markersize=.5, color='DarkGreen')
    plt.xlabel(str(nsteps) + ' points')
        
########################################################################
def make_A(k):
    """Create the matrix for the temperature problem on a k-by-k grid.
    Parameters: 
      k: number of grid points in each dimension.
    Outputs:
      A: the sparse k**2-by-k**2 matrix representing the finite difference approximation to Poisson's equation.
    """
    # First make a list with one triple (row, column, value) for each nonzero element of A
    triples = []
    for i in range(k):
        for j in range(k):
            # what row of the matrix is grid point (i,j)?
            row = j + i*k
            # the diagonal element in this row
            triples.append((row, row, 4.0))
            # connect to left grid neighbor
            if j > 0:
                triples.append((row, row - 1, -1.0))
            # ... right neighbor
            if j < k - 1:
                triples.append((row, row + 1, -1.0))
            # ... neighbor above
            if i > 0:
                triples.append((row, row - k, -1.0))
            # ... neighbor below
            if i < k - 1:
                triples.append((row, row + k, -1.0))
    
    # Finally convert the list of triples to a scipy sparse matrix
    ndim = k*k
    rownum = [t[0] for t in triples]
    colnum = [t[1] for t in triples]
    values = [t[2] for t in triples]
    A = sparse.csr_matrix((values, (rownum, colnum)), shape = (ndim, ndim))
    
    return A 

########################################################################
def make_b(k, top = 0, bottom = 0, left = 0, right = 0):
    """Create the right-hand side for the temperature problem on a k-by-k grid.
    Parameters: 
      k: number of grid points in each dimension.
      top: list of k values for top boundary (optional, defaults to 0)
      bottom: list of k values for bottom boundary (optional, defaults to 0)
      left: list of k values for top boundary (optional, defaults to 0)
      right: list of k values for top boundary (optional, defaults to 0)
    Outputs:
      b: the k**2 element vector (as a numpy array) for the rhs of the Poisson equation with given boundary conditions
    """
    # Start with a vector of zeros
    ndim = k*k
    b = np.zeros(shape = ndim)
    
    # Fill in the four boundaries as appropriate
    b[0        : k       ] += top
    b[ndim - k : ndim    ] += bottom
    b[0        : ndim : k] += left
    b[k-1      : ndim : k] += right
    
    return b
    
########################################################################
def radiator(k, width = .3, temperature = 100.):
    """Create one wall with a radiator
    Parameters: 
      k: number of grid points in each dimension; length of the wall.
      width: width of the radiator as a fraction of length of the wall (defaults to 0.2)
      temperature: temperature of the radiator (defaults to 100)
    Outputs:
      wall: the k element vector (as a numpy array) for the boundary conditions at the wall
    """
    rad_start = int(k * (0.5 - width/2))
    rad_end = int(k * (0.5 + width/2))
    wall = np.zeros(k)
    wall[rad_start : rad_end] = temperature
    
    return wall

########################################################################
def LUfactor(A, pivoting = True):
    """Factor a square matrix with partial pivoting, A[p,:] == L @ U
    Parameters: 
      A: the matrix.
      pivoting: whether or not to do partial pivoting
    Outputs (in order):
      L: the lower triangular factor, same dimensions as A, with ones on the diagonal
      U: the upper triangular factor, same dimensions as A
      p: the permutation vector that permutes the rows of A by partial pivoting
    """
    
    # Check the input
    m, n = A.shape
    assert m == n, 'input matrix A must be square'
    
    # Initialize p to be the identity permutation
    p = np.array(range(n))
    
    # Make a copy of the matrix that we will transform into L and U
    LU = A.astype(np.float64).copy()
    
    # Eliminate each column in turn
    for piv_col in range(n):
        
        # Choose the pivot row and swap it into place
        if pivoting:
            piv_row = piv_col + np.argmax(LU[piv_col:, piv_col]) 
            assert LU[piv_row, piv_col] != 0., "can't find nonzero pivot, matrix is singular"
            LU[[piv_col, piv_row], :]  = LU[[piv_row, piv_col], :]
            p[[piv_col, piv_row]]      = p[[piv_row, piv_col]]
            
        # Update the rest of the matrix
        pivot = LU[piv_col, piv_col]
        assert pivot != 0., "pivot is zero, can't continue"
        for row in range(piv_col + 1, n):
            multiplier = LU[row, piv_col] / pivot
            LU[row, piv_col] = multiplier
            LU[row, (piv_col+1):] -= multiplier * LU[piv_col, (piv_col+1):]
            
    # Separate L and U in the result
    U = np.triu(LU)
    L = LU - U + np.eye(n)
    
    return (L, U, p)

########################################################################
def Lsolve(L, b):
    """Forward solve a unit lower triangular system Ly = b for y
    Parameters: 
      L: the matrix, must be square, lower triangular, with ones on the diagonal
      b: the right-hand side vector
    Output:
      y: the solution vector to L @ y == b
    """
    
    # Check the input
    m, n = L.shape
    assert m == n, "matrix L must be square"
    assert np.all(np.tril(L) == L), "matrix L must be lower triangular"
    assert np.all(np.diag(L) == 1), "matrix L must have ones on the diagonal"
    
    # Make a copy of b that we will transform into the solution
    y = b.astype(np.float64).copy()
    
    # Forward solve
    for col in range(n):
        y[col+1:] -= y[col] * L[col+1:, col]
        
    return y

########################################################################
def Usolve(U, y):
    """Backward solve an upper triangular system Ux = y for x
    Parameters: 
      U: the matrix, must be square, upper triangular, with nonzeros on the diagonal
      y: the right-hand side vector
    Output:
      x: the solution vector to U @ x == y
    """
    
    # Check the input
    m, n = U.shape
    assert m == n, "matrix U must be square"
    assert np.all(np.triu(U) == U), "matrix U must be lower triangular"
    assert np.all(np.diag(U) != 0), "matrix U must have nonzeros on the diagonal"
    
    # Make a copy of y that we will transform into the solution
    x = y.astype(np.float64).copy()
    
    # Back solve
    for col in reversed(range(n)):
        x[col]  /= U[col, col]
        x[:col] -= x[col] * U[:col, col]
        
    return x

########################################################################
def LUfactor(A, pivoting = True):
    """Factor a square matrix with partial pivoting, A[p,:] == L @ U
    Parameters: 
      A: the matrix.
      pivoting: whether or not to do partial pivoting, default True
    Outputs (in order):
      L: the lower triangular factor, same dimensions as A, with ones on the diagonal
      U: the upper triangular factor, same dimensions as A
      p: the permutation vector that permutes the rows of A by partial pivoting
    """
    
    # Check the input
    m, n = A.shape
    assert m == n, 'input matrix A must be square'
    
    # Initialize p to be the identity permutation
    p = np.array(range(n))
    
    # Make a copy of the matrix that we will transform into L and U
    LU = A.astype(np.float64).copy()
    
    # Eliminate each column in turn
    for piv_col in range(n):
        
        # Choose the pivot row and swap it into place
        if pivoting:
            piv_row = piv_col + np.argmax(np.abs(LU[piv_col:, piv_col]))
            assert LU[piv_row, piv_col] != 0., "can't find nonzero pivot, matrix is singular"
            LU[[piv_col, piv_row], :]  = LU[[piv_row, piv_col], :]
            p[[piv_col, piv_row]]      = p[[piv_row, piv_col]]
            
        # Update the rest of the matrix
        pivot = LU[piv_col, piv_col]
        assert pivot != 0., "pivot is zero, can't continue"
        for row in range(piv_col + 1, n):
            multiplier = LU[row, piv_col] / pivot
            LU[row, piv_col] = multiplier
            LU[row, (piv_col+1):] -= multiplier * LU[piv_col, (piv_col+1):]
            
    # Separate L and U in the result
    U = np.triu(LU)
    L = LU - U + np.eye(n)
    
    return (L, U, p)

########################################################################
