#include "mkl.h"
#include <math.h>
#include <iostream>
#include <string>

char mode = 'N';
int maxIter = 10;
int debug = 1;


void printVector(std::string s, int n, const double *v) {
  std::cout << s << "\n";
  for(int i = 0; i < n; i++) {
    std::cout << v[i] << "\n";
  }
  std::cout << "\n";
}

void printNumber(std::string s, const double v) {
  std::cout << s << "\n";
  std::cout << v << "\n";
  std::cout << "\n";
}

void initializeDoubleZero(int n, double *v) {
  for(int i = 0; i < n; i++) {
    v[i] = 0;
  }
}



void fgmres(int nBlock, double *b, double *x, double *aval, int *acol, int *arow, double tol) {
  
  // data
  int n = nBlock * nBlock;
  double r0[n];
  double gamma[maxIter + 1];
  double v[n * maxIter];
  initializeDoubleZero(n * maxIter, v);
  // double z[n * maxIter];
  // initializeDoubleZero(n * maxIter, z);
  double w[n];
  double h[n * maxIter];
  initializeDoubleZero(n * maxIter, h);
  
  double beta;
  double c[maxIter + 1];
  double s[maxIter + 1];
  
  
  
  if(debug) printVector("x_0", n, x);
  if(debug) printVector("b", n, b);
  
  
  
  // start for restart loop has to be here
  
  
  // compute r0 = b-Ax_0
  mkl_cspblas_dcsrgemv(&mode , &n , aval , arow , acol , x , r0);
  if(debug) printVector("Ax_0", n, r0);
  
  vdSub(n, b, r0, r0);
  if(debug) printVector("r0", n, r0);
  
  
  //compute gamma0
  gamma[0] = cblas_dnrm2(n, r0, 1);
  if(debug) printNumber("gamma", gamma[0]);
  
  //compute v0
  cblas_dcopy (n, r0, 1, v, 1);
  cblas_dscal(n, (1 / gamma[0]), v, 1);
  if(debug) printVector("v", n, v);
  
  
//  for(int j = 0; j < maxIter; j++) {
// decreased loop for debug purposes
  for(int j = 0; j < 1; j++) {
  
    // compute q saved in w
    // here will be the conditioning in the future
    mkl_cspblas_dcsrgemv(&mode , &n , aval , arow , acol , &v[j * n] , w);
    if(debug) printVector("q", n, w);  
    
    // compute h_i,j
    for(int i = 0; i <= j; i++) {
      h[i + (j * n)] = cblas_ddot(n, &v[i * n], 0, w, 0);
      if(debug) printVector("h", n, &h[i + (j * n)]);
    }
    
    // TODO replace by mkl functions !!
    // comupte w_j
    double sum[n];
    initializeDoubleZero(n, sum);
    double tempV[n];
    for(int i = 0; i <= j; i++) {
     
      cblas_dcopy (n, &v[i * n], 1, tempV, 1);
      cblas_dscal(n, h[i + (j * n)], tempV, 1);
      
      vdAdd(n, sum, tempV, sum);
    }
    vdSub(n, w, sum, w);
    if(debug) printVector("w", n, w);
    
    //compute h_j+1,j
    h[j + 1 + (j * n)] = cblas_dnrm2(n, w, 1);
    if(debug) printNumber("h_j+1,j", h[j + 1 + (j * n)]);
    
    
    //rotation
    for(int i = 0; i < j; i++) {      
      h[i + (j * n)] = h[i + (j * n)] * c[i + 1] + h[i + 1 + (j * n)] * s[i + 1];
      h[i + 1 + (j * n)] = h[i + 1 + (j * n)] * c[i + 1] - h[i + (j * n)] * s[i + 1];
    }
    
    //compute beta
    beta = sqrt(h[j + (j * n)] * h[j + (j * n)] + h[j + 1 + (j * n)] * h[j + 1 + (j * n)]);
    
    //compute s_j+1
    s[j + 1] = h[j + 1 + (j * n)] / beta;
    
    //compute c_j+1
    c[j + 1] = h[j + (j * n)] / beta;
    
    //set h_j,j
    h[j + (j * n)] = beta;
    
    //compute gamma_j+1
    gamma[j + 1] = -(s[j + 1] * gamma[j]);
    
    //compute gamma_j
    gamma[j] = c[j + 1] * gamma[j];
    
    
    //check if finished
    if(abs(gamma[j + 1]) >= tol) {
      
      //compute v_j+1
      cblas_dcopy (n, w, 1, &v[(j + 1) * n], 1);
      cblas_dscal(n, (1 / h[j + 1 + (j * n)]), w, 1);
      break;
    } else {
    
      for(int i = j; i >= 0; i--) {
        
        
      
      }   
    }
     

  }
}