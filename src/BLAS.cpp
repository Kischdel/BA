# include <stdlib.h>
# include "blas.h"
# include "mkl.h"

namespace epsilonfem {
    

EFEM_BLAS::EFEM_BLAS(){}
        
//allocators
void* EFEM_BLAS::efem_calloc(size_t num,size_t size,int alignment) {
    return mkl_calloc(num,size,alignment);
}

void EFEM_BLAS::efem_free(void *a_ptr) {
    mkl_free(a_ptr);
}
        
//csr interface
void EFEM_BLAS::dcsrcsc(const int *job,const int *n,double *acsr,int *ja,int *ia,double *acsc,int *ja1,int *ia1,int *info) {
    mkl_dcsrcsc(job,n,acsr,ja,ia,acsc,ja1,ia1,info);
}
        
void EFEM_BLAS::dcsrgemv(const char *transa,const int *m,const double *a,const int *ia,const int *ja,const double *x,double *y) {
    mkl_cspblas_dcsrgemv(transa,m,a,ia,ja,x,y);
}
        
//vector interface
void EFEM_BLAS::dscal(const int n,const double a,double *x,const int incx) {
    cblas_dscal(n,a,x,incx);
}
        
void EFEM_BLAS::daxpy(const int n,const double a,const double *x,const int incx,double *y,const int incy) {
    cblas_daxpy(n,a,x,incx,y,incy);
}
        
void EFEM_BLAS::dswap(const int n,double *x,const int incx,double *y,const int incy) {
    cblas_dswap(n,x,incx,y,incy);
}
        
void EFEM_BLAS::dcopy(const int n,const double *x,const int incx,double *y,const int incy) {
    cblas_dcopy(n,x,incx,y,incy);
}
        
double EFEM_BLAS::ddot(const int n,const double *x,const int incx,const double *y,const int incy) {
    return cblas_ddot(n,x,incx,y,incy);
}
        
double EFEM_BLAS::dnrm2(const int n,const double *x,const int incx) {
    return cblas_dnrm2(n,x,incx);
}

void EFEM_BLAS::dcsrmultcsr(const char *trans,const int *request,const int *sort,const int *m,const int *n,const int *k,double *a,int *ja,int *ia,double *b,int *jb,int *ib,double *c,int *jc,int *ic,const int *nzmax,int *info){    
     mkl_dcsrmultcsr(trans,request,sort,m,n,k,a,ja,ia,b,jb,ib,c,jc,ic,nzmax,info);
}

}
