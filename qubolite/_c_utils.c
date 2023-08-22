#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <numpy/random/bitgen.h>
#include <numpy/random/distributions.h>
#include <omp.h>

typedef unsigned char bit;

bitgen_t *B = 0;


void print_bits(bit *x, size_t n) {
    for (size_t i=0; i<n; ++i)
        printf("%d ", x[i]);
    printf("\n");
}

double qubo_score(double **qubo, bit *x, const size_t n) {
    double v = 0.0;
    for (size_t i=0; i<n; ++i) {
        if (x[i]<=0)
            continue;
        v += qubo[i][i];
        for (size_t j=i+1; j<n; ++j)
            v += x[j]*qubo[i][j];
    }
    return v;
}


double qubo_score_flip(double **qubo, bit *x, const size_t n, const size_t i) {
    const double b = 1-2*x[i];
    double v = qubo[i][i];
    size_t j;
    for (j=0; j<i; ++j)
        v += x[j] * qubo[j][i];
    for (j=i+1; j<n; ++j)
        v += x[j] * qubo[i][j];
    return b*v;
}


struct _brute_force_result {
    bit *min_x;
    double min_val0;
    double min_val1;
} typedef brute_force_result;

brute_force_result _brute_force(
        double **qubo,
        const size_t n,
        size_t n_fixed_bits) {
    bit* x = (bit*)malloc(n*sizeof(bit)); // bit vector
    memset(x, 0, n);

    // fix some bits
    const size_t thread_id = omp_get_thread_num();
    for (size_t k=0; k<n_fixed_bits; ++k)
        x[n-k-1] = (thread_id & (1ULL<<k))>0 ? 1 : 0;
    double val = qubo_score(qubo, x, n); // QUBO value
    double dval; // QUBO value update

    bit *min_x = (bit*) malloc(n);
    memcpy(min_x, x, n);
    double min_vals[2] = {val, INFINITY};

    size_t i, j;
    // make sure it_lim is correctly set
    int64_t it_lim = (1ULL<<n-n_fixed_bits)-1;
    for (int64_t it=0; it<it_lim; ++it) {
        // get next bit flip index (gray code)
#ifdef _MSC_VER
        i = _tzcnt_u64(~it);
#else
        i = __builtin_ctzll(~it);
#endif
        x[i] ^= 1; // flip bit
        // calculate function value offset
        dval = 0;
        for (j=0; j<i; ++j)
            dval += x[j]*qubo[j][i];
        dval += qubo[i][i];
        for (j=i+1; j<n; ++j)
            dval += x[j]*qubo[i][j];

        // add or subtract, depending on bit change
        val += x[i] ? dval : -dval;
        // memorize two lowest values and bit vector
        // with lowest value (minimum solution)
        if (val<min_vals[1])
            if (val<min_vals[0]) {
                min_vals[1] = min_vals[0];
                min_vals[0] = val;
                memcpy(min_x, x, n);
            } else if (val>min_vals[0])
                min_vals[1] = val;
    }
    brute_force_result res = {min_x, min_vals[0], min_vals[1]};
    free(x);
    return res;
}

PyObject *py_brute_force(PyObject *self, PyObject *args) {
    const size_t MAX_THREADS = omp_get_max_threads();
    PyArrayObject *arr;
    size_t max_threads = MAX_THREADS;
    PyArg_ParseTuple(args, "O|k", &arr, &max_threads);
    if (PyErr_Occurred() || !PyArray_Check(arr))
	    return NULL;

    const size_t n = PyArray_DIM(arr, 0);
    double **qubo;
    npy_intp dims[] = { [0] = n, [1] = n };
    PyArray_AsCArray((PyObject**) &arr, &qubo, dims, 2,
        PyArray_DescrFromType(NPY_DOUBLE));
    if (PyErr_Occurred())
        return NULL;

#ifdef _MSC_VER
    size_t m = 63-__lzcnt64(max_threads); // floor(log2(MAX_THREADS))
#else
    size_t m = 63-__builtin_clzll(max_threads); // floor(log2(MAX_THREADS))
#endif

    // ensure that the number of bits to optimize is positive
    if (n<=m) m = n-1;

    // check if n is too large and would cause an
    // overflow of size64_t
    if (n-m>=64) {
    	// return None
	    Py_INCREF(Py_None);
	    return Py_None;
    }

    const size_t M = 1ULL<<m; // first power of 2 less or equals MAX_THREADS
    omp_set_dynamic(0);
    brute_force_result* ress = (brute_force_result*)malloc(M*sizeof(brute_force_result));
    #pragma omp parallel num_threads(M)
    {
        ress[omp_get_thread_num()] = _brute_force(qubo, n, m);
    }

    // collect all min values (except first result)
    double* all_vals = (double*)malloc((2*M-2)*sizeof(double));
    for (size_t j=1; j<M; ++j) {
        all_vals[2*j-2] = ress[j].min_val0;
        all_vals[2*j-1] = ress[j].min_val1;
    }

    // sort again to get two lowest values
    double global_min_val0 = ress[0].min_val0;
    double global_min_val1 = ress[0].min_val1;
    size_t global_min_ix = 0;
    for (size_t j=0; j<2*M-2; ++j) {
        if (all_vals[j]<global_min_val1) {
            if (all_vals[j]<global_min_val0) {
                global_min_val1 = global_min_val0;
                global_min_val0 = all_vals[j];
                global_min_ix = (j>>1)+1;
            } else if (all_vals[j]>global_min_val0) {
                global_min_val1 = all_vals[j];
            }
        }
    }

    // prepare return values
    PyObject *min_x_obj = PyArray_SimpleNew(1, &n, NPY_DOUBLE);
    double *min_x_obj_data = PyArray_DATA((PyArrayObject*) min_x_obj);
    bit *global_min_x = ress[global_min_ix].min_x;
    for (size_t j=0; j<n; ++j)
        min_x_obj_data[j] = (double) global_min_x[j];
    for (size_t j=0; j<M; ++j)
        free(ress[j].min_x);
    free(ress);
    PyObject *min_val0_obj = PyFloat_FromDouble(global_min_val0);
    PyObject *min_val1_obj = PyFloat_FromDouble(global_min_val1);
    PyObject *tup = PyTuple_New(3);
    PyTuple_SetItem(tup, 0, min_x_obj);
    PyTuple_SetItem(tup, 1, min_val0_obj);
    PyTuple_SetItem(tup, 2, min_val1_obj);
    if (PyErr_Occurred())
        return NULL;
    return tup;
}


/* ################################################
 * Gibbs sampling
 * ################################################ */

void _gibbs_sample(const size_t n, double **qubo, bit *state, size_t burn_in) {

    for (size_t i=0; i<n*burn_in; ++i) { 
        size_t v = i % n;

        double p0 = 0; // qubo_score(qubo, state, n);
        double p1 = qubo_score_flip(qubo, state, n, v); // + p0;

        if( state[v] != 0 ){
            const double temp = p0;
            p0 = p1;
            p1 = temp;
        }

        p0 = exp(-p0);
        p1 = exp(-p1);
        const double inv_Z = 1.0/(p0+p1);
        //p0 *= invZ;
        p1 *= inv_Z;

        const double u = ((double)(unsigned int)rand())/(double)UINT_MAX;
        state[v] = u < p1 ? 1 : 0;
    }
    return;
}

PyObject *py_gibbs_sample(PyObject *self, PyObject *args) {
    const size_t MAX_THREADS = omp_get_max_threads();
    PyArrayObject *arr;
    size_t max_threads = MAX_THREADS;
    size_t num_samples = 1;
    size_t burn = 100;
    size_t keep = 100;
    PyObject *bitgencap = Py_None;
    PyArg_ParseTuple(args, "O|kkkkO", &arr, &num_samples, &burn, &max_threads, &keep, &bitgencap);
    if (PyErr_Occurred() || !PyArray_Check(arr))
            return NULL;

    // max_threads = min(num_samples, max_threads);
    max_threads = 1; // don't know (yet) how to draw random numbers in a threadsafe way..

    // gen default bitgen when no bitgencapsule is provided
    if(bitgencap == Py_None) {
        PyObject *nprand = PyImport_ImportModule( "numpy.random" );
        PyObject *get_bit_generator = PyObject_GetAttrString(nprand, "get_bit_generator");

        PyObject *pyBitgen = PyObject_CallNoArgs(get_bit_generator);
        bitgencap = PyObject_GetAttrString(pyBitgen, "capsule");
    }

    bitgen_t* random_engine = (bitgen_t*) PyCapsule_GetPointer(bitgencap, "BitGenerator");

    size_t seeds[max_threads];
    bool burned[max_threads];
    for (size_t i=0; i<max_threads; ++i) {
        seeds[i] = random_uint(random_engine);
        burned[i] = false;
    }

    const size_t n = PyArray_DIM(arr, 0);
    double **qubo;
    npy_intp dims[] = { [0] = n, [1] = n };
    PyArray_AsCArray((PyObject**) &arr, &qubo, dims, 2,
        PyArray_DescrFromType(NPY_DOUBLE));
    if (PyErr_Occurred())
        return NULL;

    bit* samples = (bit*)malloc(sizeof(bit)*num_samples*n);
    memset(samples, 0, num_samples*n*sizeof(bit));

    bit* chain_state = (bit*)malloc(sizeof(bit)*max_threads*n);
    for (size_t i=0; i<max_threads*n; ++i)
        chain_state[i] = (bit) (random_uint(random_engine) % 2);

//    omp_set_dynamic(0);
// #pragma omp parallel for
    for (size_t j=0; j<num_samples; ++j) {
        const size_t tid = omp_get_thread_num();
	_gibbs_sample(n, qubo, chain_state+(tid*n), burned[tid] ? keep : burn);
        memcpy(samples+(j*n), chain_state+(tid*n), sizeof(bit)*n);
    }

    free(chain_state);

    npy_intp rdims[] = { [0] = num_samples, [1] = n };
    PyObject* res = PyArray_SimpleNewFromData(2, rdims, NPY_UINT8, samples);
    Py_INCREF(res);
    return res;
}

/* ################################################
 * Python module def                              
 * ################################################ */

static PyMethodDef methods[] = {
    {"brute_force", py_brute_force, METH_VARARGS, "Solves QUBO the hard way"},
    {"gibbs_sample", py_gibbs_sample, METH_VARARGS, "Sample from the induced exponential family"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_c_utils",
    NULL, -1, methods
};

PyMODINIT_FUNC PyInit__c_utils() {
    import_array();
    return PyModule_Create(&module);
}
