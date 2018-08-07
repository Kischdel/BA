#pragma once
namespace epsilonfem {
    class EFEM_BLAS {
    private:
        
    public:
        //constructors
        EFEM_BLAS();
        
        //allocators
        static void* efem_calloc (size_t num, size_t size, int alignment);
        static void efem_free (void *a_ptr);
        
        //csr interface
        static void dcsrcsc (const int *job , const int *n , double *acsr , int *ja , int *ia , double *acsc , int *ja1 , int *ia1 , int *info );
        static void dcsrgemv (const char *transa , const int *m , const double *a , const int *ia , const int *ja , const double *x , double *y );
        
        //vector interface
        static void dscal (const int n, const double a, double *x, const int incx);
        static void daxpy (const int n, const double a, const double *x, const int incx, double *y, const int incy);
        static void dswap (const int n, double *x, const int incx, double *y, const int incy);
        static void dcopy (const int n, const double *x, const int incx, double *y, const int incy);
        static double ddot (const int n, const double *x, const int incx, const double *y, const int incy);
        static double dnrm2 (const int n, const double *x, const int incx);

	//matrix interface
	static void dcsrmultcsr (const char *trans, const int *request, const int *sort, const int *m, const int *n, const int *k, double *a, int *ja, int *ia, double *b, int *jb, int *ib, double *c, int *jc, int *ic, const int *nzmax, int *info);
        
    };    
    
}
