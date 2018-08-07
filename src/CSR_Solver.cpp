# include <cmath>
# include <iostream> 
# include <cstring>
# include <stdexcept>
# include <chrono>
# include "CSR_Solver.h"
# include "blas.h"
# include "jacobi_par.h"

namespace epsilonfem {
    
//constructor
CSR_Solver::CSR_Solver(){}
        
//interface

        
//right preconditioned GMRES with saved preconditioners for flexible changes, also with restart option        
int CSR_Solver::FGMRES(SparseMatrix *A, double *b, double *x0, double tol, int restart, double *result, int preIter) {
/*
    *Incomming parameters:
    *A -> left hand side
    *b -> right hand side
    *x0 -> initial guess of solution
    *tol -> ||x||2<=tol to be solution
    *restart -> restart after k steps
    *result -> solution
    */
    int debug = 1;
    const char transpose = 'n';
    
//prepare arrays
    //std::vector<int> *ia = A->getI();
    //std::vector<int> *ja = A->getJ();
    //std::vector<double> *data = A->getData();
    
    //const int vectorsize = b->size();
    //const int size_data = data->size();
    //const int size_ia = ia->size();
    //const int size_ja = ja->size();
    //const int rows = A->getRows();
    //const int cols = A->getCols();
    
    // my matrix objects use a different data layout
    const int vectorsize = A->n;
    const int size_data = A->valSize;
    const int size_ia = A->rowSize;
    const int size_ja = A->valSize;
    const int rows = A->n;
    const int cols = A->n;
    
    
    if (rows!=cols) {
        throw std::invalid_argument("Solver breakdown. Matrix must be quadratic.");
    }
    if (rows!=vectorsize || size_ja!=size_data) {
        throw std::invalid_argument("Solver breakdown. Missmatched dimensions in used matrix or vectors.");
    }
    
    int *matrix_i = (int *) EFEM_BLAS::efem_calloc(size_ia,sizeof(int),64);
    int *matrix_j = (int *) EFEM_BLAS::efem_calloc(size_ja,sizeof(int),64);
    double *matrix_a = (double *) EFEM_BLAS::efem_calloc(size_data,sizeof(double),64);
    double *vec_b = (double *) EFEM_BLAS::efem_calloc(vectorsize,sizeof(double),64);
    double *vec_x0 = (double *) EFEM_BLAS::efem_calloc(vectorsize,sizeof(double),64);
    
    //std::memcpy(matrix_i,&(ia->at(0)),sizeof(int)*size_ia);
    //std::memcpy(matrix_j,&(ja->at(0)),sizeof(int)*size_data);
    //std::memcpy(matrix_a,&(data->at(0)),sizeof(double)*size_data);
    //std::memcpy(vec_b,&(b->at(0)),sizeof(double)*vectorsize);
    //std::memcpy(vec_x0,&(x0->at(0)),sizeof(double)*vectorsize);
    
    std::memcpy(matrix_i,A->row,sizeof(int)*size_ia);
    std::memcpy(matrix_j,A->col,sizeof(int)*size_data);
    std::memcpy(matrix_a,A->val,sizeof(double)*size_data);
    std::memcpy(vec_b,b,sizeof(double)*vectorsize);
    std::memcpy(vec_x0,x0,sizeof(double)*vectorsize);
    
    
//first step preperations
    int reached_tolerance = 0;
    int doRestart = 0;
    int number_of_restarts = 0;
        
    const int min_memory = restart>vectorsize ? vectorsize : restart;
        
    int stepcounter = 0;
        
//allocate memory
    double *h = (double *) EFEM_BLAS::efem_calloc((min_memory+1) * min_memory,sizeof(double),64); //initial hessenberg matrix
    double *g = (double *) EFEM_BLAS::efem_calloc(min_memory+1,sizeof(double),64);	              //tolerance test    
    double *rot_sin = (double *) EFEM_BLAS::efem_calloc(min_memory,sizeof(double),64);	      //rotation values for sine rotation
    double *rot_cos = (double *) EFEM_BLAS::efem_calloc(min_memory,sizeof(double),64);	      //rotation values for cosine rotation
    double *y = (double *) EFEM_BLAS::efem_calloc(min_memory,sizeof(double),64);	              //coefficients to compute solution
    double *r = (double *) EFEM_BLAS::efem_calloc(vectorsize,sizeof(double),64);	              //residual
    double *v = (double *) EFEM_BLAS::efem_calloc(vectorsize*(min_memory+1),sizeof(double),64);       //vectors to compute solution
    double *w = (double *) EFEM_BLAS::efem_calloc(vectorsize,sizeof(double),64);	              //helper for hessenberg computation
    double *z = (double *) EFEM_BLAS::efem_calloc(vectorsize*min_memory,sizeof(double),64);	      //helper for preconditioning
    double *vec_result = (double *) EFEM_BLAS::efem_calloc(vectorsize,sizeof(double),64);	      //temporary memory for result

    double *s = (double *) EFEM_BLAS::efem_calloc(vectorsize,sizeof(double),64);	              //helper for preconditioning
    
    
    
//messuring execution time
    std::vector<std::chrono::high_resolution_clock::time_point> jacobi_start;
    std::vector<std::chrono::high_resolution_clock::time_point> jacobi_stop;
    
    std::chrono::high_resolution_clock::time_point FGMRES_start = std::chrono::high_resolution_clock::now();
    

//first step computations
    EFEM_BLAS::dcsrgemv(&transpose, &vectorsize, matrix_a, matrix_i, matrix_j, vec_x0, r);   //residual
    EFEM_BLAS::dscal (vectorsize, -1, r, 1);
    EFEM_BLAS::daxpy (vectorsize, 1, vec_b, 1, r, 1);
    g[0] = EFEM_BLAS::dnrm2 (vectorsize, r, 1);                                  //tolerance test
    EFEM_BLAS::dcopy (vectorsize, r, 1, &v[0], 1);                               //vectors to compute solution
    EFEM_BLAS::dscal (vectorsize, 1.0/g[0], &v[0], 1);
    
//iterations until convergence or doRestart
    while(reached_tolerance==0 && doRestart==0) {
        
//check for bad preconditioner        
        if (stepcounter>=vectorsize) {
            throw std::invalid_argument("Solver breakdown. Bad preconditioner warning. Needs >N steps to solve.");
        }
//preconditioning  
        //P->precondition(vectorsize,&v[stepcounter*vectorsize],&z[stepcounter*vectorsize]);
        
        //messure time
        jacobi_start.push_back(std::chrono::high_resolution_clock::now());
        
        jacobiLower(A, preIter, &v[stepcounter*vectorsize], s);
        jacobiUpper(A, preIter, s, &z[stepcounter*vectorsize]);
        
        jacobi_stop.push_back(std::chrono::high_resolution_clock::now());
        
//arnoldi process
        EFEM_BLAS::dcsrgemv(&transpose, &vectorsize, matrix_a, matrix_i, matrix_j, &z[stepcounter*vectorsize], w);  
        
//computing hessenberg matrix
        for (int i=0;i<=stepcounter;++i) {
            h[i*min_memory+stepcounter] = EFEM_BLAS::ddot (vectorsize, &v[i*vectorsize], 1, w, 1);
        }
        
        for (int i=0;i<=stepcounter;++i) {
            EFEM_BLAS::daxpy (vectorsize,(-1.0)*h[i*min_memory+stepcounter],&v[i*vectorsize],1,w,1);
        }
        h[(stepcounter+1)*min_memory+stepcounter] = EFEM_BLAS::dnrm2 (vectorsize, w, 1); 

//rotations
        for (int i=0;i<stepcounter;++i) {
            const double tmp = h[i*min_memory+stepcounter];
            h[i*min_memory+stepcounter] = (rot_cos[i] * h[i*min_memory+stepcounter]) + (rot_sin[i] * h[(i+1)*min_memory+stepcounter]);
            h[(i+1)*min_memory+stepcounter] = (rot_sin[i] * tmp * (-1)) + (rot_cos[i] * h[(i+1)*min_memory+stepcounter]);
        }
   
        double beta = sqrt(pow(h[stepcounter*min_memory+stepcounter],2) + pow(h[(stepcounter+1)*min_memory+stepcounter],2)); 
        rot_sin[stepcounter] = h[(stepcounter+1)*min_memory+stepcounter]/beta;
        rot_cos[stepcounter] = h[stepcounter*min_memory+stepcounter]/beta;                
        h[stepcounter*min_memory+stepcounter] = beta;
        
//computing residual
        g[stepcounter+1] = (-1.0)*rot_sin[stepcounter]*g[stepcounter];
        g[stepcounter] = rot_cos[stepcounter]*g[stepcounter];
        
//         std::cout  << fabs(g[stepcounter+1]) << std::endl;
        
//computing coefficients
        if (fabs(g[stepcounter+1])>=tol) {
            if (stepcounter+1>=restart) {doRestart = 1;}
            else {
                EFEM_BLAS::dcopy (vectorsize, w, 1, &v[(stepcounter+1)*vectorsize], 1); 
                EFEM_BLAS::dscal (vectorsize, 1.0/h[(stepcounter+1)*min_memory+stepcounter], &v[(stepcounter+1)*vectorsize], 1);
                stepcounter++;
            }
        }
        else {
            reached_tolerance = 1;
            EFEM_BLAS::dscal (min_memory, 0, y, 1);
            for (int i=stepcounter;i>=0;--i) {
                for(int k=i+1;k<=stepcounter;++k) {
                    y[i] = y[i] + (h[i*min_memory+k] * y[k]);
                }
                y[i] = 1.0/h[i*min_memory+i] * (g[i] - y[i]);
            }
        }

//return to top, doRestart, or finish
        if (doRestart==1) {
            number_of_restarts++;                                       //reset flags an memory
            doRestart = 0;
            EFEM_BLAS::dscal (min_memory, 0, y, 1);
            EFEM_BLAS::dscal (vectorsize, 0, vec_result, 1);

            for (int i=stepcounter;i>=0;--i) {				//compute coefficients for x_m
                for(int k=i+1;k<=stepcounter;++k) {
                    y[i] = y[i] + (h[i*min_memory+k] * y[k]);
                }
                y[i] = 1.0/h[i*min_memory+i] * (g[i] - y[i]);
            }
	
            for (int i=0;i<=stepcounter;++i) {                          //compute x_m
                EFEM_BLAS::daxpy (vectorsize, y[i], &z[i*vectorsize], 1, vec_result, 1);
            }
            EFEM_BLAS::daxpy (vectorsize, 1, vec_x0, 1, vec_result, 1);
           
            EFEM_BLAS::dcsrgemv(&transpose, &vectorsize, matrix_a, matrix_i, matrix_j, vec_result, r);//new residual
            EFEM_BLAS::dscal (vectorsize, -1, r, 1);
            EFEM_BLAS::daxpy (vectorsize, 1, vec_b, 1, r, 1);
            
            EFEM_BLAS::dcopy (vectorsize, vec_result, 1, vec_x0, 1);                                         //x_m -> x_0
            
            g[0] = EFEM_BLAS::dnrm2 (vectorsize, r, 1);                                                      //tolerance test
            
            EFEM_BLAS::dcopy (vectorsize, r, 1, &v[0], 1);                                                   //vectors to compute solution
            EFEM_BLAS::dscal (vectorsize, 1.0/g[0], &v[0], 1);

            stepcounter = 0;			
        }
    }
    
//compute approximate solution
    EFEM_BLAS::dscal (vectorsize, 0, vec_result, 1);

    for (int i=0;i<=stepcounter;++i) {
       EFEM_BLAS::daxpy (vectorsize, y[i], &z[i*vectorsize], 1, vec_result, 1);
    }
    EFEM_BLAS::daxpy (vectorsize, 1, vec_x0, 1, vec_result, 1);
    
    for (int i=0;i<vectorsize;++i) {
            result[i] = vec_result[i];
        }
    
    /*
    int rSize = (int)result->size();
    
    if (rSize>=vectorsize) {
        std::memcpy(&(result->at(0)),&vec_result[0],sizeof(double)*vectorsize);
    }
    else {
        result->clear();
        for (int i=0;i<vectorsize;++i) {
            result->push_back(vec_result[i]);
        }
    }
    */
    
//end messure time
    std::chrono::high_resolution_clock::time_point FGMRES_stop = std::chrono::high_resolution_clock::now();
  
    auto duration_FGMRES = std::chrono::duration_cast<std::chrono::microseconds>(FGMRES_stop - FGMRES_start).count();
    
    std::cout << "execution time: " << duration_FGMRES / 1000000 << " seconds \n";
    
    int steps = restart*number_of_restarts + stepcounter + 1;
    
    auto duration_jacobi = 0;
    
    for (int i = 0; i < jacobi_start.size(); i++) {
      auto diff = std::chrono::duration_cast<std::chrono::microseconds>(jacobi_stop.at(i) - jacobi_start.at(i)).count();
      duration_jacobi += diff;
    }
    std::cout << "jacobi time: " << (double)duration_jacobi / 1000000 << " seconds \n";
    std::cout << "average jacobi time: " << ((double)duration_jacobi / steps) / 1000000 << " seconds \n";
    
    
//test solution
    if (debug == 1) {
        EFEM_BLAS::dcsrgemv(&transpose, &vectorsize, matrix_a, matrix_i, matrix_j, vec_result, r);
        EFEM_BLAS::daxpy (vectorsize, -1, vec_b, 1, r, 1);
        double sp = EFEM_BLAS::dnrm2 (vectorsize, r, 1); 
        std::cout << "Analysis of result -> " << sp << std::endl; 
	
        int steps = restart*number_of_restarts + stepcounter + 1;
        std::cout << "Finished after " << steps << " steps (" << number_of_restarts << " restarts)" << std::endl;
    }

    EFEM_BLAS::efem_free(g);
    EFEM_BLAS::efem_free(h);
    EFEM_BLAS::efem_free(y);
    EFEM_BLAS::efem_free(rot_sin);
    EFEM_BLAS::efem_free(rot_cos);
    EFEM_BLAS::efem_free(r);
    EFEM_BLAS::efem_free(v);
    EFEM_BLAS::efem_free(w);
    EFEM_BLAS::efem_free(z);
    EFEM_BLAS::efem_free(vec_b);
    EFEM_BLAS::efem_free(vec_x0);
    EFEM_BLAS::efem_free(vec_result);
    EFEM_BLAS::efem_free(matrix_a);
    EFEM_BLAS::efem_free(matrix_i);
    EFEM_BLAS::efem_free(matrix_j);
    
    EFEM_BLAS::efem_free(s);
        
    return 0;
}
    
}
