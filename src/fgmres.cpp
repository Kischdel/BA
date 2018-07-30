#include "mkl.h"
#include <math.h>
#include <iostream>
#include <string>

char mode = 'N';
int maxIter = 10;
int debug = 0;
int verbose = 0;
int fgmresLog = 1;


void printMatrix(std::string s, int n, int m, const double *v) {
  std::cout << s << "\n";
  for(int i = 0; i < n; i++) {
    for(int j = 0; j < m; j++) {
    
      std::cout << v[i + (j * n)] << "    ";
    }
     std::cout << "\n";
  }
  std::cout << "\n";
}

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
  
  
  // ============= INIT ================ //
  // data
  int n = nBlock * nBlock;
  double r0[n];
  double gamma[maxIter + 1];
  double v[n * maxIter];
  initializeDoubleZero(n * maxIter, v);
  // double z[n * maxIter];
  // initializeDoubleZero(n * maxIter, z);
  
  double w[n];
  int nH = maxIter + 1;
  double h[nH * maxIter];
  
  double beta;
  double c[maxIter + 1];
  double s[maxIter + 1];
  double y[maxIter];
  
  int running = 1; //is set to 0 when tol is satisfied and disables restart
  int restartCounter = 0; // for logging purposes
  
  if(debug) printVector("debug: x_0", n, x);
  if(debug) printVector("debug: b", n, b);
  
  // ========================= //
  // start of the restart loop //
  // ========================= //
  do{
  
    //clear old valus
    initializeDoubleZero(nH * maxIter, h);
    initializeDoubleZero(n * maxIter, v);
    // initializeDoubleZero(n * maxIter, z);
  
    // compute r0 = b-Ax_0
    mkl_cspblas_dcsrgemv(&mode , &n , aval , arow , acol , x , r0);
    if(fgmresLog) std::cout << "THIS IS RESTART # " << restartCounter << "\n \n";
    if(debug) printVector("debug: Ax_0", n, r0);
  
    vdSub(n, b, r0, r0);
    if(debug) printVector("debug: r0", n, r0);
  
  
    //compute gamma0
    gamma[0] = cblas_dnrm2(n, r0, 1);
    if(debug) printNumber("debug: gamma_0", gamma[0]);
    printNumber("debug: gamma_0", gamma[0]);
  
    //compute v0
    cblas_dcopy (n, r0, 1, v, 1);
    cblas_dscal(n, (1.0 / gamma[0]), v, 1);
    if(debug) printVector("debug: v", n, v);
  
  
    for(int j = 0; j < maxIter; j++) {
  
      // compute q saved in w
      // here will be the conditioning in the future
      mkl_cspblas_dcsrgemv(&mode , &n , aval , arow , acol , &v[j * n] , w);
      if(debug) printVector("debug: q", n, w);  
    
      // compute h_i,j
      for(int i = 0; i <= j; i++) {
        h[i + (j * nH)] = cblas_ddot(n, &v[i * n], 0, w, 0);
      }
      if(verbose) printMatrix("verbose: h", nH, maxIter, h);
          
      // comupte w_j
      double sum[n];
      initializeDoubleZero(n, sum);
      double tempV[n];
      for(int i = 0; i <= j; i++) {
     
        cblas_dcopy (n, &v[i * n], 1, tempV, 1);
        cblas_dscal(n, h[i + (j * nH)], tempV, 1);
      
        vdAdd(n, sum, tempV, sum);
      }
      vdSub(n, w, sum, w);
      if(debug) printVector("debug: w", n, w);
    
      //compute h_j+1,j
      h[j + 1 + (j * nH)] = cblas_dnrm2(n, w, 1);
      if(debug) printNumber("debug: h_j+1,j", h[j + 1 + (j * nH)]);
      if(verbose) printMatrix("verbose: h", nH, maxIter, h);
    
      //rotation
      for(int i = 0; i < j; i++) {      
        h[i + (j * nH)] = h[i + (j * nH)] * c[i + 1] + h[i + 1 + (j * nH)] * s[i + 1];
        h[i + 1 + (j * nH)] = h[i + 1 + (j * nH)] * c[i + 1] - h[i + (j * nH)] * s[i + 1];
      }
      if(verbose) printMatrix("verbose: h after rotation", nH, maxIter, h);
      
      
      //compute beta
      beta = sqrt(h[j + (j * nH)] * h[j + (j * nH)] + h[j + 1 + (j * nH)] * h[j + 1 + (j * nH)]);
      if(debug) printNumber("debug: beta", beta);
    
      //compute s_j+1
      s[j + 1] = h[j + 1 + (j * nH)] / beta;
      if(debug) printNumber("debug: s_j+1", s[j + 1]);
    
      //compute c_j+1
      c[j + 1] = h[j + (j * nH)] / beta;
      if(debug) printNumber("debug: c_j+1", c[j + 1]);
    
      //set h_j,j
      h[j + (j * nH)] = beta;
      if(debug) printNumber("debug: h_j,j", h[j + (j * nH)]);
      
      //compute gamma_j+1
      gamma[j + 1] = -(s[j + 1] * gamma[j]);
      if(debug) printNumber("debug: gamma_j+1", gamma[j + 1]);
    
      //compute gamma_j
      gamma[j] = c[j + 1] * gamma[j];
      if(debug) printNumber("debug: gamma_j", gamma[j]);
    
    
      if(debug) printMatrix("debug: h befor tolcheck", nH, maxIter, h);
      
      if(fgmresLog) {
        std::cout << "j: " << j << " and maxIter: " << maxIter << " in Restart # " << restartCounter << "\n";
        std::cout << "tol is: " << tol << " and fabs(gamma[j + 1]): " << fabs(gamma[j + 1]) << "\n \n";  
      }
      
      //check if finished or maxIter reached
      if(fabs(gamma[j + 1]) >= tol && j < maxIter - 1) {
      
        //compute v_j+1
        cblas_dcopy (n, w, 1, &v[(j + 1) * n], 1);
        cblas_dscal(n, (1.0 / h[j + 1 + (j * nH)]), &v[(j + 1) * n], 1);
        if(debug) printVector("debug: v_j+1", n, &v[(j + 1) * n]);
        
        if(fgmresLog) std::cout << "NEXT ROUND WITH j++ \n \n";
        
      } else {
        
        if(fgmresLog) std::cout << "ELSE ENTERED \n \n";
        
        
        for(int i = j; i >= 0; i--) {
          
          double sum = 0;
          for(int k = i + 1; k <= j; k++) {
            sum += h[i + (k * nH)] * y[k];
          }
          
          y[i] = (gamma[i] - sum) / h[i + (i * nH)];
              
        }
        
        initializeDoubleZero(n, sum);
        
        for(int i = 0; i <= j; i++) {
     
          cblas_dcopy (n, &v[i * n], 1, tempV, 1);
          cblas_dscal(n, y[i], tempV, 1);
      
          vdAdd(n, sum, tempV, sum);
        }
        
        if(debug) printVector("debug: delta x", n, sum);
        
        vdAdd(n, x, sum, x);
        
        if(debug) printVector("debug: new x", n, x);        
        
        if(fabs(gamma[j + 1]) < tol) running = 0;
        break;
      }
    } // end forloop to maxIter
    
    if(fgmresLog) {
        std::cout << "this message is printed, if a restart occurs. if running is 0 the algorithm has finished. running: " << running << "\n"; 
    }
    
    if(running) restartCounter++;
    
  } while (running && restartCounter < 4);
  //END
}