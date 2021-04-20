
import getopt
import importlib
import os
import re
import sys
import traceback

import numpy as np
# import scipy as sp
import scipy.linalg as la
import scipy.integrate
import h5py

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class DOF:

    """
    A global degree of freedom.

    Each degree of freedom is identified by an arbitrary number, which
    may be used to enumerate the scalar FE equations.

    Geometric boundary conditions, including prescribed displacements,
    may be imposed at each DOF.
    """

    def __init__(self, number, bc):
        """
        Create an object representing a global degree of freedom.

        Arguments
            number  Integer identifying the DOF (i.e., equation number)
            bc      Geometric boundary condtion.

            bc may be any of None, meaning the degree of freedom
            is unconstrained; a scalar (e.g., 0.0), which will be
            imposed at all times; or a function of time (e.g.,
            harmonic motion).

            If bc is provided as a function, it should take one argument,
            a scalar or 1-D array t of time values, and return
            displacment, velocity, and acceleration values of the same
            shape as t.
        """
        self.number = number
        self.bc = bc


class Node:

    """
    A node in the finite element mesh.

    Each node has an arbitrary identifying number, a unique location,
    and a global degree of freedom.
    """

    def __init__(self, number, x, dof):
        """
        Create an object representing a node.

        Arguments
            number  Integer identifying the node
            x       Global coordinate
            dof     DOF object to be associated with this node
        """
        self.number = number
        self.x = x
        self.dof = dof


class Element:

    """
    A finite element representing a portion of the domain.
    """

    def __init__(self, number, nodes):
        """
        Create an object representing an element.

        Arguments
            number  Integer identifying the element
            nodes   List of 2 Nodes defining the element

            The DOF of the nodes will be accessible through this
            object's dof attribute.  NB: These are not simply numbers.
        """
        self.number = number
        self.nodes = nodes
        self.dof = [self.nodes[0].dof, self.nodes[1].dof]
        self.length = self.nodes[1].x - self.nodes[0].x

    def mass_matrix(self):
        """
        Return the mass matrix of this element.

        Returns
            m  Element mass matrix (2 x 2 NumPy array)
        """
        #m = (self.length / 6.0) * np.array([[2.0, 1.0],
        #                                    [1.0, 2.0]])
        l=self.length
        Cm = 1.0   # rho.A
        # m = np.array([[156, 22*l, 54, -13*l],
		# 		  [22*l, 4*l*l, 13*l, -3*l*l],
		# 		  [54, 13*l, 156, -22*l],
		# 		  [-13*l, -3*l*l, -22*l, 4*l*l]]) * Cm * l / 420
        m = np.array([[156,  54],				  
				      [54, 156]]) * Cm * l / 420
        return m

    def damping_matrix(self, lam):
        """
        Return the damping matrix of this element.

        Argument
            lam  Distributed damping coefficient (scalar)

        Returns
            c  Element damping matrix (2 x 2 NumPy array)
        """
        #c = lam * (self.length / 6.0) * np.array([[2.0, 1.0],
        #                                          [1.0, 2.0]])
        l=self.length
        # c = np.array([[156, 22*l, 54, -13*l],
		# 		  [22*l, 4*l*l, 13*l, -3*l*l],
		# 		  [54, 13*l, 156, -22*l],
		# 		  [-13*l, -3*l*l, -22*l, 4*l*l]])  * lam * l / 420
        c = np.array([[156, 54],
				     [54,  156], ])  * lam * l / 420
        
        return c

    def stiffness_matrix(self):
        """
        Return the linear stiffness matrix of this element.

        Returns
            k  Element stiffness matrix (2 x 2 NumPy array)
        """
        #k = np.array([[ 1.0, -1.0],
        #              [-1.0,  1.0]]) / self.length
        l=self.length
        Ck = 1.0   # E.I
        # k = np.array([[12, 6*l, -12, 6*l],
		# 		  [6*l, 4*l*l, -6*l, 2*l*l],
		# 		  [-12, -6*l, 12, -6*l],
		# 		  [6*l, 2*l*l, -6*l, 4*l*l]]) * Ck / l**3
        k = np.array([[12, -12],				  
				     [-12, 12] ]) * Ck / l**3
        
        return k


class Problem:

    """
    Collection of functions defining an FEA run.
    """

    def __init__(self, linear, ke0, pe0, lam, t_max, dt, x,
                 bcs, ic_shapes, ic_modes):
        """
        Arguments
            linear     Problem type (Boolean)
            ke0        Desired initial kinetic energy (scalar)
            pe0        Desired initial potential energy (scalar)
            lam        Distributed damping coefficient (scalar)
            t_max      Duration of analysis (scalar)
            dt         Time step (scalar)
            x          Coordinates of nodes (1-D NumPy array)
            bcs        Left and right-end BCs (list of 2 scalars or fns of t)
            ic_shapes  Initial displacement and velocity distributions
                       (list of two 1-D NumPy arrays)
            ic_modes   Modes used to represent ICs (list of ints); if empty,
                       no modal projection is done

            BCs can be specified as either displaements (scalars) or
            functions of time (a scalar or 1-D NumPy array) returning
            displacement, velocity, and acceleration (each the same
            shape as t).  Such functions can be defined within this
            function.
        """
        self.linear = linear
        self.ke0 = ke0
        self.pe0 = pe0
        self.lam = lam
        self.t_max = t_max
        self.dt = dt
        self.x = x
        self.bcs = bcs
        self.ic_shapes = ic_shapes
        self.ic_modes = ic_modes


class Solution:

    """
    Collected results of an FEA run.
    """

    def __init__(self, prob, u0, u_t0, t, u, u_t):
        self.linear = prob.linear
        self.ke0 = prob.ke0
        self.pe0 = prob.pe0
        self.lam = prob.lam
        self.t_max = prob.t_max
        self.dt = prob.dt
        self.x = prob.x
        self.u0 = u0
        self.u_t0 = u_t0
        self.t = t
        self.u = u
        self.u_t = u_t

    def postprocess(self, M, K, Phi):
        self.M = M
        self.K = K
        self.Phi = Phi
        self.ke = np.array([kinetic_energy(self.u_t[:,i], self.M)
                            for i in range(self.u_t.shape[1])])
        self.pe = np.array([potential_energy(self.u[:,i], self.K, self.linear)
                            for i in range(self.u.shape[1])])

    # def save(self, overwrite=True):
    #     """
    #     Write a solution to an HDF5 file.
    #     """
    #     hdf5_file_name = gen_hdf5_file_name(self.name)
    #     if os.path.exists(hdf5_file_name):
    #         if not overwrite:
    #             emsg('File\n' +
    #                  '{}\n'.format(hdf5_file_name) +
    #                  'exists and overwrite is False.\n')
    #             sys.exit(1)
    #         else:
    #             vmsg(verbose,
    #                   'Removing existing file {}.\n'.format(hdf5_file_name))
    #             os.unlink(hdf5_file_name)
    #     smsg('Saving solution to file {}.\n'.format(hdf5_file_name))
    #     with h5py.File(hdf5_file_name, 'w') as hdf5:
    #         hdf5.attrs['name'] = prob.name
    #         hdf5.attrs['num_nodes'] = num_nodes
    #         hdf5.attrs['lam'] = prob.lam
    #         hdf5.attrs['ke0'] = prob.ke0
    #         hdf5.attrs['pe0'] = prob.pe0
    #         hdf5.attrs['t_max'] = prob.t_max
    #         hdf5.attrs['dt'] = prob.dt
    #         hdf5['x'] = prob.x
    #         hdf5['u0'] = u0
    #         hdf5['u_t0'] = u_t0
    #         hdf5['u'] = u
    #         hdf5['u_t'] = u_t
    #         hdf5['ke'] = ke
    #         hdf5['pe'] = pe
    #         hdf5['Phi'] = Phi
    #
    # def load(self):
    #     """
    #     Read a solution from an HDF5 file.
    #     """
    #     hdf5_file_name = gen_hdf5_file_name(self.name)
    #     if not os.path.exists(hdf5_file_name):
    #         emsg('File\n' +
    #              '{}\n'.format(hdf5_file_name) +
    #              'does not exist.\n')
    #         sys.exit(1)
    #     smsg('Loading solution from file {}.\n'.format(hdf5_file_name))
    #     with h5py.File(hdf5_file_name, 'r') as hdf5:
    #         self.name = hdf5.attrs['name']
    #         self.num_nodes = hdf5.attrs['num_nodes']
    #         self.lam = hdf5.attrs['lam']
    #         self.ke0 = hdf5.attrs['ke0']
    #         self.pe0 = hdf5.attrs['pe0']
    #         self.t_max  = hdf5.attrs['t_max']
    #         self.dt = hdf5.attrs['dt']
    #         self.x = hdf5['x'][...]
    #         self.u0 = hdf5['u0'][...]
    #         self.u_t0 = hdf5['u_t0'][...]
    #         self.u = hdf5['u'][...]
    #         self.u_t = hdf5['u_t'][...]
    #         self.ke = hdf5['ke'][...]
    #         self.pe = hdf5['pe'][...]
    #         self.Phi = hdf5['Phi'][...]
    #     # Reconstruct the time vector, which was not stored.
    #     self.t = np.arange(self.u.shape[1]) * self.dt

    def plot_mesh(self):
        """
        Plot the FE mesh.
        """
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        plt.plot(self.x, np.zeros(self.x.shape), '-bo')
        plt.xlim((self.x.min(), self.x.max()))
        plt.grid()
        plt.title('Mesh')
        plt.xlabel('$x$')
        xmargin = 0.1 * (self.x[-1] - self.x[0])
        v = np.array(plt.axis())
        v[0] = self.x[0] - xmargin
        v[1] = self.x[-1] + xmargin
        plt.axis(v)
        plt.yticks([])
        plt.connect('key_press_event', key_press_event)

    def plot_ics(self):
        """
        Plot the initial conditions of the string.
        """
        fig = plt.figure()
        fig.patch.set_facecolor('white')

        plt.subplot(2,1,1)
        plt.plot(self.x, self.u0)
        plt.xlim((self.x.min(), self.x.max()))
        plt.grid()
        plt.title('Initial displacement')
        plt.xlabel('$x$')
        plt.ylabel('$u_{0}$')

        plt.subplot(2,1,2)
        plt.plot(self.x, self.u_t0)
        plt.xlim((self.x.min(), self.x.max()))
        plt.grid()
        plt.title('Initial velocity')
        plt.xlabel('$x$')
        plt.ylabel('$\dot{u}_{0}$')

        plt.connect('key_press_event', key_press_event)

    def plot_ctr_disp_vel(self):
        """
        Plot displacement and velocity at the center of the string.
        """
        idx = int(round(self.u.shape[0]/2))

        fig = plt.figure()
        fig.patch.set_facecolor('white')

        plt.subplot(2,1,1)
        plt.plot(self.t, self.u[idx, :])
        plt.xlim((self.t.min(), self.t.max()))
        plt.grid()
        plt.title('String displacement at center')
        plt.xlabel('$t$')
        plt.ylabel('$u$')

        plt.subplot(2,1,2)
        plt.plot(self.t, self.u_t[idx, :])
        plt.xlim((self.t.min(), self.t.max()))
        plt.grid()
        plt.title('String velocity at center')
        plt.xlabel('$t$')
        plt.ylabel('$u_{t}$')

        plt.connect('key_press_event', key_press_event)

    def plot_disp_snaps(self):
        """
        Plot snapshots of the displacement.
        """
        num_curves = 21
        # T1 = t[-1]
        T1 = 2.0 / 5.7
        stride = int(round(T1 / self.dt / num_curves))

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        # plt.hold(True)
        idx = len(self.t) - 1
        for i in range(num_curves):
            plt.plot(self.x, self.u[:, idx])
            idx -= stride
        plt.xlim((self.x.min(), self.x.max()))
        plt.grid()
        plt.title('String displacement snapshots')
        plt.xlabel('$x$')
        plt.ylabel('$u$')
        plt.connect('key_press_event', key_press_event)

    def plot_vel_snaps(self):
        """
        Plot snapshots of the velocity.
        """
        num_curves = 21
        T1 = self.t[-1]
        stride = int(round(T1 / self.dt / num_curves))

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        # plt.hold(True)
        idx = len(self.t) - 1
        for i in range(num_curves):
            plt.plot(self.x, self.u_t[:, idx])
            idx -= stride
        plt.xlim((self.x.min(), self.x.max()))
        plt.grid()
        plt.title('String velocity snapshots')
        plt.xlabel('$x$')
        plt.ylabel('$u_{t}$')
        plt.connect('key_press_event', key_press_event)

    def plot_disp_surf(self):
        """
        Plot displacement as a surface over (x,t).
        """
        X, T = np.meshgrid(self.x, self.t)

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(T, X, self.u.T, rstride=10, cstride=10)
        ax.set_title('String displacement')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        plt.connect('key_press_event', key_press_event)

    def plot_vel_surf(self):
        """
        Plot velocity as a surface over (x,t).
        """
        X, T = np.meshgrid(self.x, self.t)

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(T, X, self.u_t.T, rstride=10, cstride=10)
        ax.set_title('String velocity')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        plt.connect('key_press_event', key_press_event)

    def plot_energy_th(self):
        """
        Plot energy time histories.
        """
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        # plt.hold(True)
        plt.plot(self.t, self.ke, 'k')
        plt.plot(self.t, self.pe, 'b')
        plt.plot(self.t, self.ke + self.pe, 'r')
        plt.legend(('KE', 'PE', 'Total'))
        plt.xlim((self.t.min(), self.t.max()))
        plt.grid()
        plt.title('Energy')
        plt.xlabel('$t$')
        plt.ylabel('$E$')
        plt.connect('key_press_event', key_press_event)

    def plot_ke_xt(self):
        """
        Plot kinetic energy density as an x-t diagram.
        """
        skip = 0
        X, T = np.meshgrid(self.x, self.t[skip:])
        KE = 0.5 * self.u_t[:,skip:]**2

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        plt.pcolormesh(X, T, KE.T, shading='auto')
        # plt.colorbar()
        plt.title('String kinetic energy density')
        plt.xlabel('$x$')
        plt.ylabel('$t$')
        plt.connect('key_press_event', key_press_event)

    def plot_ke_surf(self):
        """
        Plot kinetic energy density as a surface over (x,t).
        """
        X, T = np.meshgrid(self.x, self.t)
        KE = 0.5 * self.u_t.T**2

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(T, X, KE, rstride=10, cstride=10)
        ax.set_title('String kinetic energy density')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        plt.connect('key_press_event', key_press_event)

    def plot_disp_proj(self):
        """
        Plot modal projections of displacement.
        """
        u_proj = np.abs(self.Phi.T.dot(self.u))
        n = np.arange(u_proj.shape[0])
        N, T = np.meshgrid(n, self.t)

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        plt.pcolormesh(T, N, u_proj.T, shading='auto')
        # plt.colorbar()
        plt.title('Modal projections of displacement')
        plt.xlabel('$t$')
        plt.ylabel('$n$')
        plt.connect('key_press_event', key_press_event)

    def plot_vel_proj(self):
        """
        Plot modal projections of velocity.
        """
        u_t_proj = np.abs(self.Phi.T.dot(self.u_t))
        n = np.arange(u_t_proj.shape[0])
        N, T = np.meshgrid(n, self.t)

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        plt.pcolormesh(T, N, u_t_proj.T, shading='auto')
        # plt.colorbar()
        plt.title('Modal projections of velocity')
        plt.xlabel('$t$')
        plt.ylabel('$n$')
        plt.connect('key_press_event', key_press_event)

    def plot_disp_asym(self):
        """
        Plot the asymmetry of the solution as a surface over (x,t).
        """
        u_asymmetry = self.u[:,:] - self.u[-1::-1,:]
        X, T = np.meshgrid(self.x, self.t)

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(T, X, u_asymmetry.T)
        ax.set_title('String displacement asymmetry')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        plt.connect('key_press_event', key_press_event)

    def plot_vel_asym(self):
        """
        Plot the asymmetryt of the velocity as a surface over (x,t).
        """
        u_t_asymmetry = self.u_t[:,:] - self.u_t[-1::-1,:]
        X, T = np.meshgrid(self.x, self.t)

        fig = plt.figure()
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.plot_surface(T, X, u_t_asymmetry.T)
        ax.set_title('String velocity asymmetry')
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        plt.connect('key_press_event', key_press_event)


def simulate(prob):
    verbose = True
    overwrite = True
    smsg('Simulation beginning.\n')

    dofs, nodes, elts, N, N1, N2, idx1, idx2, num_nodes = gen_mesh(prob)
    M, C, K = global_matrices(N, elts, prob.lam)
    M11, M12, M21, M22 = partition_global_matrix(M, idx1, idx2)
    C11, C12, C21, C22 = partition_global_matrix(C, idx1, idx2)
    K11, K12, K21, K22 = partition_global_matrix(K, idx1, idx2)

    evals, evecs = la.eig(K11, M11)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:,idx]
    omega = np.sqrt(np.real(evals))
    Phi = np.zeros((N,N1))
    for i in range(len(evals)):
        Phi[idx1,i] = evecs[:,i]
        Phi[:,i] /= np.sqrt(Phi[:,i].dot(M.dot(Phi[:,i])))
    if verbose:
        print('Lowest frequencies of linear string:')
        print(omega[:5])

    u0, u_t0 = prob.ic_shapes
    if len(prob.ic_modes) > 0:
        vmsg(verbose, 'Projecting initial conditions onto selected modes.\n')
        Phi_hat = Phi[:,prob.ic_modes]
        u0 = np.dot(Phi_hat, Phi_hat.T.dot(u0))
        u_t0 = np.dot(Phi_hat, Phi_hat.T.dot(u_t0))

    vmsg(verbose, 'Scaling initial conditions.\n')
    if prob.pe0 > 0.0:
        u0 = u0 * (prob.pe0 / potential_energy(u0, K, prob.linear))**(1/4)
    else:
        u0 = np.zeros(u0.shape)
    if prob.ke0 > 0.0:
        u_t0 = u_t0 * np.sqrt(prob.ke0 / kinetic_energy(u_t0, M))
    else:
        u_t0 = np.zeros(u_t0.shape)

    vmsg(verbose, 'Starting time integration.\n')
    t = np.arange(0.0, prob.t_max, prob.dt)
    y0 = np.zeros(2*N1)
    y0[:N1] = u0[idx1]
    y0[N1:] = u_t0[idx1]
    M11LU = la.lu_factor(M11)

    def eom(y, t):
        y1 = y[:N1]
        y2 = y[N1:]
        u2, u2_t, u2_tt = boundary_conditions(t, dofs, idx2)
        if prob.linear:
            alpha = 1.0
        else:
            u = np.zeros(N)
            u[idx1] = y1
            u[idx2] = u2
            alpha = u.dot(K.dot(u))
        yt = np.zeros(y.shape)
        yt[:N1] = y2
        yt[N1:] = la.lu_solve(M11LU,
                    - C11.dot(y2) - alpha * K11.dot(y1)
                    - M12.dot(u2_tt) - C12.dot(u2_t) - alpha * K12.dot(u2))
        return yt

    y = scipy.integrate.odeint(eom, y0, t)

    vmsg(verbose, 'Extracting u and u_t from y and inserting BCs.\n')
    u2, u2_t, u2_tt = boundary_conditions(t, dofs, idx2)
    u = np.zeros((N, len(t)))
    u_t = np.zeros((N, len(t)))
    u[idx1,:] = y.T[:N1]
    u_t[idx1,:] = y.T[N1:]
    u[idx2,:] = u2[...]
    u_t[idx2,:] = u2_t[...]

    vmsg(verbose, 'Creating Solution instance from time histories.\n')
    soln = Solution(prob, u0, u_t0, t, u, u_t)
    smsg('Simulation completed.\n')

    smsg('Postprocessing.\n')
    soln.postprocess(M, K, Phi)
    smsg('Postprocessing completed.\n')
    return soln


def gen_hdf5_file_name(name):
    hdf5_file_name = '{}.hdf5'.format(name)
    return hdf5_file_name


def gen_mesh(prob):
    """
    Return nodes, DOF, and elements.

    Arguments
        prob  Problem instance

    Returns
        dofs   List of DOF instances
        nodes  List of Node instances
        elts   List of Element instances
        N      Number of degrees of freedom (total)
        N1     Number of unconstrained degrees of freedom
        N2     Number of constrained degrees of freedom (BCs)
        idx1   Scalar eq numbers (constrained) (1-D NumPy arrary)
        idx2   Scalar eq numbers (unconstrained) (1-D NumPy arrary)
    """
    num_nodes = len(prob.x)
    num_dofs = num_nodes
    num_elts = num_nodes - 1

    dofs = []
    for i in range(num_dofs):
        dofs.append(DOF(i, None))
    for i, bc in prob.bcs:
        dofs[i].bc = bc

    nodes = []
    for i in range(num_nodes):
        nodes.append(Node(i, prob.x[i], dofs[i]))

    elts = []
    for i in range(num_elts):
        elts.append(Element(i, [nodes[i], nodes[i+1]]))

    idx1 = []
    idx2 = []
    for dof in dofs:
        if dof.bc is None:
            idx1.append(dof.number)
        else:
            idx2.append(dof.number)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    N = len(dofs)
    N1 = len(idx1)
    N2 = len(idx2)

    return dofs, nodes, elts, N, N1, N2, idx1, idx2, num_nodes


def global_matrices(N, elts, lam):
    """
    Return global mass, damping, and linear stiffness matrices.

    Arguments
        N         Total number of degrees of freedom
        elts      List of Element instances
        lam       Distributed damping coefficient (scalar)

    Returns
        M  Global mass matrix (N x N NumPy array)
        C  Global damping matrix (N x N NumPy array)
        K  Global linear stiffness matrix (N x N NumPy array)
    """
    M = np.zeros((N,N))
    C = np.zeros((N,N))
    K = np.zeros((N,N))
    for elt in elts:
        add_to_global_matrix(M, elt.mass_matrix(), elt.dof)
        add_to_global_matrix(C, elt.damping_matrix(lam), elt.dof)
        add_to_global_matrix(K, elt.stiffness_matrix(), elt.dof)
    return M, C, K


def add_to_global_matrix(A, a, dof):
    """
    Add contributions of element matrix a to global matrix A.

    Arguments
        A    Global matrix (N x N NumPy array)
        a    Element matrix (2 x 2 NumPy array)
        dof  List of 2 DOF instances
    """
    n0, n1 = [dof[i].number for i in (0, 1)]
    A[n0,n0] += a[0,0]
    A[n0,n1] += a[0,1]
    A[n1,n0] += a[1,0]
    A[n1,n1] += a[1,1]


def partition_global_matrix(A, idx1, idx2):
    """
    Return partitions of global matrix A.

    Arguments
        A     Global matrix (N x N NumPy array)
        idx1  Indices of unconstrained scalar eqs (1-D NumPy array)
        idx2  Indices of constrained scalar eqs (1-D NumPy array)

    Returns
        A11  Partition of A indexed by (idx1, idx1)
        A12  Partition of A indexed by (idx1, idx2)
        A21  Partition of A indexed by (idx2, idx1)
        A22  Partition of A indexed by (idx2, idx2)
    """
    A11 = A[idx1,:][:,idx1]
    A12 = A[idx1,:][:,idx2]
    A21 = A[idx2,:][:,idx1]
    A22 = A[idx2,:][:,idx2]
    return A11, A12, A21, A22


def boundary_conditions(t, dofs, idx2):
    """
    Return the values of the geometric boundary conditions.

    Arguments
        t     Time
        dofs  List of DOF instances where BCs are applied
        idx2  Indices of constrained scalar eqs (1-D NumPy array)

    Returns
        u2     Boundary displacements
        u2_t   Boundary velocities
        u2_tt  Boundary accelerations

        If t is a vector (1-D NumPy array), each of these is a 1-D
        NumPy array of size N2 x len(t). If t is a scalar, each of
        these is a 1-D NumPy array of length N2.
    """
    if np.isscalar(t):
        u2 = np.zeros(len(idx2))
    else:
        u2 = np.zeros((len(idx2), len(t)))
    u2_t = np.zeros(u2.shape)
    u2_tt = np.zeros(u2.shape)
    for i, idx in enumerate(idx2):
        if callable(dofs[idx].bc):
            g, g_t, g_tt = dofs[idx].bc(t)
        else:
            g = dofs[idx].bc * np.ones(np.size(t))
            g_t = np.zeros(np.shape(t))
            g_tt = np.zeros(np.shape(t))
        u2[i] = g
        u2_t[i] = g_t
        u2_tt[i] = g_tt
    return u2, u2_t, u2_tt


def kinetic_energy(u_t, M):
    """
    Return the kinetic energy of the continuum.返回连续体的动能

    Arguments
        u_t  Velocity vector (1-D NumPy array)
        M    Global mass matrix (2-D NumPy array)

    Returns
        ke  Kinetic energy (scalar)
    """
    ke = u_t.dot(M.dot(u_t)) / 2.0
    return ke


def potential_energy(u, K, linear):
    """
    Return the potential energy of the continuum.

    Arguments
        u       Displacement vector (1-D NumPy array)
        K       Global linear stiffness matrix
        linear  Problem type (Boolean)

    Returns
        pe  Potential energy (scalar)
    """
    if linear:
        pe = u.dot(K.dot(u)) / 2.0
    else:
        pe = u.dot(K.dot(u))**2 / 4.0
    return pe


def key_press_event(event):
    """Interpret a single keypress in a plot window."""
    if event.key == 'c':
        plt.close()
    elif event.key == 'q':
        plt.close('all')


def emsg(s):
    """Display an error message."""
    sys.stderr.write('{}: {}\n'.format(sys.argv[0], s))

def smsg(s):
    """Display a status message."""
    vmsg(True, s)

def vmsg(verbose, s):
    """Display a status message if being verbose."""
    if verbose:
        sys.stdout.write(s)