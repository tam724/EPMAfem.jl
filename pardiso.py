class PanuaPardiso:
    '''
    This is an interface to the Panua Pardiso linear solver library.

    https://panua.ch/pardiso/

    The interface supports float(64) and int(32).

    There are three parts to the solution of a linear system:

    1. Symbolic factorization: Analyzes the structure of the nonzero
       pattern. This has to be done before anything else can be done.
    2. Numerical facotorization: Factorizes the matrix, using the
       information from the symbolic factorization.
    3. Solving a linear system:  After the numerical factorization has
       taken place, linear systems can be solved.

    In addition, this interface gives access to a special featue of
    Panua Pardiso. With the "selected inversion" one can compute values
    of the inverse of the matrix at the position of the nonzeros of the
    original matrix.

    The matrix must be provided in csr (compressed sparse row) format.
    For symmtric matrices, internally, only the upper triangular part
    must be given, and nonzeros entries must exist for all diagonal
    elements (even if their values are zero). One can accellerate the
    computation by providing the input matrix already in this form.
    Otherwise, the matrix is internally modified accordingly.
    '''

    def __init__(self, mtype=11, verbose=False,
                 is_upper_triangular=False, has_full_diagonal=False,
                 skip_matrix_check=False, libdir=None):
        '''
        Initialization of a PanuaPardiso object.  It loads the C
        functions from a shared library and initializes Pardiso.

        Parameters
        ----------

        mtype : int
            Matrix type for the matrices to be handled by this object:
                 1: real and structurally symmetric
                 2: real and symmetric positive defnite
                -2: real and symmetric indefnite
                11: real and nonsymmetric
        verbose : bool
            If True, Pardiso will write its diagnostic output to stdout.
        is_upper_triangular : bool
            For symmetric matrices only.
            If True, only the upper triangular part of the symmetric
            input matrix is provided.  Otherwise, an internal copy is
            formed with the expected structure.
        has_full_diagonal: bool
            For symmetric matrices only.
            If True, all diagonal elements are included in the nonzero
            structure of the input matrix, even for entries with a zero
            value.  Otherwise, an internal copy is formd with the
            expected structure.
        skip_matrix_check : bool
            By default, Pardiso's pardiso_chkmatrix function is called
            for each input matrix as sanity check.  Once the user is
            sure that the input matrix is always properly formed, this
            can be set to True.
        libdir : str
            If this, this is the path to the direction in which the
            Pardiso library libpardiso.* is located.

        '''

        # Figure out the name of the shared object to be loaded
        libname = "libpardiso"

        # detect operating system
        os_type = platform.system()

        if os_type == "Linux":
            dirsepchar = '/'
            so_ext = '.so'
        elif os_type == "Darwin":
            dirsepchar = '/'
            so_ext = '.dylib'
        elif os_type == "Windows":
            dirsepchar = '\\'
            so_ext = '.dll'
        else:
            raise NotImplementedError(
                """This interface has not been implemented for operating
                system {}.""". format(os_type))

        # Add the shared library extension
        libname = libname+so_ext

        # Add the path if given
        if libdir != None:
            libname = libdir+dirsepchar+libname

        # Load the Pardiso solver library
        libpardiso = CDLL(libname)

        # With this flag we keep track of whether the matrix has
        # already been factorized
        self.have_symbolic_factorization = False
        self.have_factorization = False

        # We store information to tell which modifications need to be
        # made to the structure
        self._is_upper_triangular = is_upper_triangular
        self._has_full_diagonal = has_full_diagonal
        self._skip_matrix_check = skip_matrix_check

        # Define pointer types
        c_int32_p = POINTER(c_int32)
        c_int64_p = POINTER(c_int64)
        c_double_p = POINTER(c_double)

        # Initialize the Pardiso arguments that need to be kept track
        # of
        self.PT = np.zeros(64, dtype=np.int64)
        self.MTYPE = c_int32(mtype)
        self._IPARM = np.zeros(64, dtype=np.int32)
        self._DPARM = np.zeros(64, dtype=np.double)
        if verbose:
            self.MSGLVL = c_int32(1)
        else:
            self.MSGLVL = c_int32(0)
        # For symmstric matrices, we need to provide only the lower
        # triangular part
        if not mtype in [1, 2, -2]:
            self._A_is_symmtric = True
        else:
            self._A_is_symmtric = False

        # TODO: check if this is a 64bit platform.

        # Get pardisoinit function to initialize the solver.
        pardisoinit = libpardiso.pardisoinit
        pardisoinit.argtypes = [
            c_int64_p,   # PT(64)
            c_int32_p,   # MTYPE
            c_int32_p,   # SOLVER
            c_int32_p,   # IPARM
            c_double_p,  # DPARM
            c_int32_p   # ERROR
        ]

        # We are using the direct solver.
        solver = 0

        # Call pardisoinit to initialize the solver
        ERROR = c_int32()
        pardisoinit(
            self.PT.ctypes.data_as(c_int64_p),
            byref(self.MTYPE),
            byref(c_int32(solver)),
            self._IPARM.ctypes.data_as(c_int32_p),
            self._DPARM.ctypes.data_as(c_double_p),
            byref(ERROR))

        # Check if there is an error
        if ERROR.value != 0:
            # TODO: replace with proper exception
            raise "Pardiso initialization failed"

        # Get Pardiso function ready
        self.pardiso = libpardiso.pardiso
        self.pardiso.argtypes = [
            c_int64_p,   # PT(64)
            c_int32_p,   # MAXFCT
            c_int32_p,   # MNUM
            c_int32_p,   # MTYPE
            c_int32_p,   # PHASE
            c_int32_p,   # N
            c_double_p,  # A
            c_int32_p,   # IA
            c_int32_p,   # JA
            c_int32_p,   # PERM
            c_int32_p,   # NRHS
            c_int32_p,   # IPARM
            c_int32_p,   # MSGLVL
            c_double_p,  # B
            c_double_p,  # X
            c_int32_p,   # ERROR
            c_double_p   # DPARM
        ]

        self.pardiso_chkmatrix = libpardiso.pardiso_chkmatrix
        self.pardiso_chkmatrix.argtypes = [
            c_int32_p,   # MTYPE
            c_int32_p,   # N
            c_double_p,  # A
            c_int32_p,   # IA
            c_int32_p,   # JA
            c_int32_p    # ERROR
        ]

    def _call_pardiso(self, phase, A_csr, b, x):
        '''
        Internal method for calling the Pardiso C function.

        It checks the matrix unless skip_matrix_check is set to True.
        '''

        # Define pointer types
        c_int32_p = POINTER(c_int32)
        c_int64_p = POINTER(c_int64)
        c_double_p = POINTER(c_double)

        # Sanity checks
        if A_csr.shape[0] != A_csr.shape[1]:
            raise ValueError('A_csr is not a square matrix.')
        # if A_csr.shape[0] != B.shape[0]:
        #    raise ValueError("B has incorrect dimension {}, should be {}.".format(B.shape[0], A.shape[0]))
        # if A_csr.shape[0] != X.shape[0]:
        #    raise ValueError("X has incorrect dimension {}, should be {}.".format(X.shape[0], A.shape[0]))

        # Check if the data is OK
        if not self._skip_matrix_check:
            error = self.check_matrix(A_csr)
            if error != 0:
                raise ValueError("""A_csr does not have required format
                                 and has_full_diagonal or is_upper
                                 triangular is set to True.""")

        # Set local input arguments
        MAXFCT = c_int32(1)
        MNUM = c_int32(1)
        PHASE = c_int32(phase)
        N = c_int32(A_csr.shape[0])
        A = A_csr.data
        IA = A_csr.indptr+1
        JA = A_csr.indices+1
        PERM = c_int32(0)
        NRHS = c_int32(1)
        if b.ndim > 1:
            NRHS = b.shape[1]
        ERROR = c_int32()

        # Call pardiso function
        self.pardiso(
            self.PT.ctypes.data_as(c_int64_p),
            byref(MAXFCT),
            byref(MNUM),
            byref(self.MTYPE),
            byref(PHASE),
            byref(N),
            A.ctypes.data_as(c_double_p),
            IA.ctypes.data_as(c_int32_p),
            JA.ctypes.data_as(c_int32_p),
            byref(PERM),
            byref(NRHS),
            self._IPARM.ctypes.data_as(c_int32_p),
            byref(self.MSGLVL),
            b.ctypes.data_as(c_double_p),
            x.ctypes.data_as(c_double_p),
            byref(ERROR),
            self._DPARM.ctypes.data_as(c_double_p))

        return ERROR.value

    def _correct_matrix_structure(self, A_csr):
        '''
        For symmetric matrices, we need to make sure that they have
        only upper-triangular values and that there are elements on
        the diagonal.
        '''

        # There is nothing to do if this is not a symmtic matrix
        if not self.MTYPE.value in [1, 2, -2]:
            return A_csr

        if not self._is_upper_triangular:
            # get rid of the lower-triangular part
            A_csr = sp.sparse.triu(A_csr, format='csr')

        if not self._has_full_diagonal:
            # As a dirty trick, we add a really tiny amount to the
            # diagonal elements to make sure that they are
            # structurally present
            n = A_csr.shape[0]
            A_csr += A_csr + 1e-200 * sp.sparse.identity(n)

        return A_csr

    def check_matrix(self, A_csr):
        '''
        Call Pardiso's pardiso_chkmatrix function to verify that the
        input matrix is properly formed. It is also automatically
        called for the other methods if necessary (see constructor).
        The check_matrix() method is provided so that a user can check
        a matrix during development, before calling Pardiso.

        Parameters
        ----------

        A_csr: scipy.sparse.csr_matrix
            Input matrix to be analyzed.

        Returns
        -------

        error : int
            = 0 if matrix is properly formed.
            Otherwise, there is an issue with the matrix.  See the
            stdout output of the method for more information.
        '''

        # Define pointer types
        c_int32_p = POINTER(c_int32)
        c_double_p = POINTER(c_double)

        # Make required corrections
        A_csr = self._correct_matrix_structure(A_csr)

        # Set local input arguments
        N = c_int32(A_csr.shape[0])
        A = A_csr.data
        IA = A_csr.indptr+1
        JA = A_csr.indices+1
        ERROR = c_int32()

        # Call pardiso_chkmatrix function
        self.pardiso_chkmatrix(
            byref(self.MTYPE),
            byref(N),
            A.ctypes.data_as(c_double_p),
            IA.ctypes.data_as(c_int32_p),
            JA.ctypes.data_as(c_int32_p),
            byref(ERROR))

        return ERROR.value

    def factorize(self, A_csr, new_structure=True,
                  already_corrected=False):
        '''
        Factorize the matrix A_csr. The user must set new_structure to
        True if no factorzation or the factorization of a matrix with a
        different sparsity structure has been done before.
    
        Parameters
        ----------

        A_csr: scipy.sparse.csr_matrix
            Input matrix to be factorized.

        new_structure : bool
            If set to True, the symbolic factorization will be
            performed.  Otherwise the symbolic factorization is skipped
            (can be quite a bit faster) but the symbolic factorization
            of a matrix with the same sparsity structure must have been
            performed earlier, either with a call to factorize() or
            solve().

        already_corrected : bool
            This is ignored for non-symmetric matrices.
            For a symmetric matrix, the input matrix will be corrected
            so that it fits the internal Pardiso requirements (when
            upper_triangular = False or has_full_diagonal = False was
            given in the constructor). This correction is skipped if
            already_corrected is set to True.

        Returns
        -------

        error : int
            = 0 if matrix factorization was successful.
            Otherwise, an error occured.  Refer to the Pardiso manual
            for the different error codes.
        '''

        # should we do some sanity checks to make sure A and MTYPE
        # match?

        if not new_structure and not self.have_symbolic_factorization:
            raise ValueError(
                '''Argument same_structure set to True, but matrix has
                not been factorized before''')

        if new_structure:
            phase = 12
            # we need to make sure that the indices are sorted for
            # Pardiso
            if not A_csr.has_sorted_indices:
                raise ValueError("A_csr must have sorted indices")
            # A_csr.sort_indices()
        else:
            phase = 22

        # dummy vectors for rhs and solution
        b = np.array([0.])
        x = np.array([0.])

        # Make required corrections
        if not already_corrected:
            A_csr = self._correct_matrix_structure(A_csr)

        error = self._call_pardiso(phase, A_csr, b, x)

        # we now have the symbolic factorization
        if error == 0:
            self.have_symbolic_factorization = True
            self.have_factorization = True
        else:
            self.have_symbolic_factorization = False
            self.have_factorization = False

        return error

    def solve(self, A_csr, b, transpose = False, 
              new_matrix=True, new_structure=True):
        '''
        Solve the linear system A_csr * x = b for x.

        Several linear systems are solved simultaneously when b is a
        two-dimensional array.

        If necessary, the input matrix is factorized first.

        Parameters
        ----------

        A_csr: scipy.sparse.csr_matrix
            Coefficient matrix in the linera system.

        b : numpy.array
            Right hand side(s) for the linear system(s).  If the second
            dimension is larger than 1, several linear systems are
            solved at the same time.
        
        transpose : bool
            If set to True, the linear systems are solved with the
            transpose of the matrix.

        new_matrix : bool
            If this matrix has been factorized before(because the
            factorized() method was called before or because linear
            systems have already been solved with this matrix), this
            should be set to False.

        new_structure : bool
            This only applies then new_matrix = True.
            If set to true, the symbolic factorization will be
            performed.  Otherwise the symbolic factorization is skipped
            (quite a bit faster) but the symbolic factorization of a
            matrix with the same sparsity structure must have been done
            earlier.

        Returns
        -------

        x : numpy.array
            The solution vector(s) of the linear system(s).  This has
            the same dimensions as the input b.

        error : int
            = 0 if the linear systems were solved successfully.
            Otherwise, an error occured.  Refer to the Pardiso manual
            for the different error codes.
        '''

        if not new_matrix and not self.have_factorization:
            raise ValueError(
                '''Argument new_matrix set to False but matrix has not
                been factorized before''')

        # Make required corrections
        A_csr = self._correct_matrix_structure(A_csr)

        if new_matrix:
            error = self.factorize(A_csr, new_structure,
                                   already_corrected=True)
            if (error != 0):
                return error

        x = np.zeros_like(b)

        # Set the transpose flag
        if transpose:
            self._IPARM[11] = 1
        else:
            self._IPARM[11] = 0

        phase = 33
        error = self._call_pardiso(phase, A_csr, b, x)

        return (x, error)

    def selected_inversion(self, A_csr):
        '''
        Compute the elements of the inverse matrix of A_csr at the
        positions of the nonzero elements of A_csr.

        The factorize method much have been called before for the
        input matrix.

        Parameters
        ----------

        A_csr: scipy.sparse.csr_matrix
            Input matrix for which the elements of the inverse are
            desired.

        Returns
        -------

        A_inv : scipy.sparse,csr_matrix
            This matrix has the same structure and sparsity structure
            as A_csr. The nonzero elements of its sparity structure
            contain the elements of the inverse of A_csr at these
            position.

        error : int
            = 0 if the calculation was successful.
            Otherwise, an error occured.  Refer to the Pardiso manual
            for the different error codes.
        '''
        if not self.have_factorization:
            raise ValueError(
                '''To call selected_inversion, you need to call factorization
                first''')

        # Make required corrections
        A_csr = self._correct_matrix_structure(A_csr)

        # For this call, Pardiso overwrites the input matrix
        A_inv = A_csr.copy()

        # dummy vectors for rhs and solution
        b = np.array([0.])
        x = np.array([0.])

        # Call Pardiso to overwrite the nonzeros of A_inv with the
        # elememts of the inverse
        phase = -22
        error = self._call_pardiso(phase, A_inv, b, x)

        # we now have the symbolic factorization
        if error == 0:
            self.have_symbolic_factorization = True
            self.have_factorization = True
        else:
            self.have_symbolic_factorization = False
            self.have_factorization = False

        return A_inv, error

    def set_IPARM(self, i, val):
        '''
        Set Pardiso's IPARM flag array entry i to val. Some entries of
        the IPARM set options for Pardiso. Refer to the Pardiso
        interface for the description of IPARM.

        Parameters
        ----------

        i : int
            Index of IPARM element that should be changed.  The index
            counting starts at 1 (FORTRAN-style) in consistency with
            the Pardiso documentation.

        val : int
            Value to which the entry should be set.
        '''

        self._IPARM[i-1] = val

    def set_DPARM(self, i, val):
        '''
        Set Pardiso's DPARM flag array entry i to val. Some entries of
        DPARM set options for Pardiso. Refer to the Pardiso interface
        for the description of DPARM.

        Parameters
        ----------

        i : int
            Index of the DPARM element that should be changed. The
            index counting starts at 1 (FORTRAN-style) in consistency
            with the Pardiso documentation.

        val : double
            Value to which the entry should be set.
        '''

        self._DPARM[i-1] = val


    def get_IPARM(self, i):
        '''
        Get the i-th entry of Pardiso's IPARM array. Some entries
        contain diagnostic output after an operation. Refer to the
        Pardiso interface for the description of DPARM.

        Parameters
        ----------

        i : int
            Index of the IPARM element that should be returned. The
            index counting starts at 1 (FORTRAN-style) in consistency
            with the Pardiso documentation.

        Returns
        -------
        val : double
            Value of IPARM(i)
        '''

        return self._IPARM[i-1]
    
    def get_DPARM(self, i):
        '''
        Get the i-th entry of Pardiso's DPARM array. Some entries
        contain diagnostic output after an operation. Refer to the
        Pardiso interface for the description of DPARM.

        Parameters
        ----------

        i : int
            Index of the DPARM element that should be returned. The
            index counting starts at 1 (FORTRAN-style) in consistency
            with the Pardiso documentation.

        Returns
        -------
        val : double
            Value of DPARM(i)
        '''

        return self._DPARM[i-1]

