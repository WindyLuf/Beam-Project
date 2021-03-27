import numpy as np              #numpy                                #Eigenvalues ​​and eigenvectors
from scipy.linalg import eigh   #返回复数Hermitian(共轭对称)或实对称矩阵的特征值和特征向量
from matplotlib import pyplot as plt 
import time

def beam(num_elems):
    restrained_dofs = [1, 0]
	#restrained_dofs = [1, 0, -2, -1]
 
    l = 1.0 / num_elems
    Cm = 1.0   # rho.A
    Ck = 1.0   # E.I

	# element mass and stiffness matrices
    m = np.array([[156, 22*l, 54, -13*l],
				  [22*l, 4*l*l, 13*l, -3*l*l],
				  [54, 13*l, 156, -22*l],
				  [-13*l, -3*l*l, -22*l, 4*l*l]]) * Cm * l / 420

    k = np.array([[12, 6*l, -12, 6*l],
				  [6*l, 4*l*l, -6*l, 2*l*l],
				  [-12, -6*l, 12, -6*l],
				  [6*l, 2*l*l, -6*l, 4*l*l]]) * Ck / l**3

	# construct global mass and stiffness matrices
    M = np.zeros((2*num_elems+2,2*num_elems+2))
    K = np.zeros((2*num_elems+2,2*num_elems+2))

	# for each element, change to global coordinates
    for i in range(num_elems):
        M_temp = np.zeros((2*num_elems+2,2*num_elems+2))
        K_temp = np.zeros((2*num_elems+2,2*num_elems+2))
        M_temp[2*i:2*i+4, 2*i:2*i+4] = m
        K_temp[2*i:2*i+4, 2*i:2*i+4] = k
        M += M_temp
        K += K_temp

	# remove the fixed degrees of freedom
    for dof in restrained_dofs:
        for i in [0,1]:
            M = np.delete(M, dof, axis=i)
            K = np.delete(K, dof, axis=i)

    evals, evecs = eigh(K,M)    #返回复数Hermitian(共轭对称)或实对称矩阵的特征值和特征向量
    frequencies = np.sqrt(evals)
    return M, K, frequencies, evecs,evals


M, K, frequencies, evecs ,evals=beam(8)  #If divided into eight element
print("M=",M)
print("K=",K)
print("evals=",evals)
print("evecs=",evecs)
print("frequencies=",frequencies)




















"""
# beam element
print("Beam element")
# exact_frequency = math.pi**2   #  simply supported
# exact_frequency = 1.875104**2  #  cantilever beam
# exact_frequency = 3.926602**2  #  built in - pinned beam
exact_frequency = 4.730041**2    #  fixed-fixed

errors = []
for i in range(2,20):     # number of elements
	start = time.perf_counter()
	M, K, frequencies, evecs = beam(i)
	time_taken = time.perf_counter() - start
	error = (frequencies[0] - exact_frequency) / exact_frequency * 100.0
	errors.append( (i, error) )
	print ('Num Elems: {} \tFrequency: {}\tError: {}% \tShape: {} \tTime: {}'
                .format( i, round(frequencies[0],3), round(error, 3), K.shape, round(time_taken*1000, 3) ))

print ('Exact Freq:', round(exact_frequency, 3) )

element  = np.array([x[0] for x in errors])
error   = np.array([x[1] for x in errors])


# plot the result
plt.plot(element, error, 'o-')
plt.xlim(1, element[-1])
plt.xlabel('Number of Elements')
plt.ylabel('Errror (%)')
plt.show()
"""