#pragma once
# include <vector>
# include "sparsematrix.h"

// # include "basicPreconditioner.h"

namespace epsilonfem {
    class CSR_Solver {
    private:
        
    public:
        //constructors
        CSR_Solver();
        
        //interface
        
        /*
        int GMRES(Sparse_Matrix *A,std::vector<double> *b,std::vector<double> *x0,double tol,int restart,std::vector<double> *result);
        int PGMRES(Sparse_Matrix *A,std::vector<double> *b,std::vector<double> *x0,double tol,int restart,std::vector<double> *result);
        */
        
        int FGMRES(SparseMatrix *A, double *b, double *x0, double tol, int restart, double *result, int preIter);
    } ;   
    
}
