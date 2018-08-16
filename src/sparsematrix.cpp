#include "sparsematrix.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>

void SparseMatrix::print() {

  int nextVal = 0;
  
  for (int i = 0; i < this->n ; i++) {
    for (int j = 0; j < this->n; j++) {
    
      if (j == col[nextVal]) {
        std::cout <<  val[nextVal] << "\t";
        nextVal++;
      } else {
        std::cout << "0\t";
      }
    }
    std::cout << "\n";
  }
}

void SparseMatrix::printILU() {

  int nextVal = 0;
  
  for (int i = 0; i < this->n ; i++) {
    for (int j = 0; j < this->n; j++) {
    
      if (j == col[nextVal]) {
        std::cout <<  valILU[nextVal] << "\t";
        nextVal++;
      } else {
        std::cout << "0\t";
      }
    }
    std::cout << "\n";
  }
}

void SparseMatrix::computeILU() {
  
  for(int k = 0; k < n - 1; k++) {

    std::cout << "\nk: " << k << " / " << n - 2;
    
    for(int i = k + 1; i < k + 1 + nBlock && i < n; i += nBlock - 1) {
      std::cout << "    i/k: " << i << "/" << k << "  ";
      if(getValAt(i, k) != 0) {
        std::cout << " != 0";
        double temp = getILUValAt(i,k) / getILUValAt(k,k);
        setILUValAt(i, k, temp);
        
        // #pragma omp parallel for
        for(int j = k + 1; j <= (k + (2 * nBlock)) && j < n; j++) {
          
          if(getValAt(i, j) != 0 && getValAt(k, j) != 0) {
            
            //temp = getILUValAt(i, j) - (temp * getILUValAt(k, j));
            //temp = getILUValAt(i, j) - (getILUValAt(i, k) * getILUValAt(k, j));
            setILUValAt(i, j, getILUValAt(i, j) - (temp * getILUValAt(k, j)));
            
          }
        }
      }
    }
  }
}


void SparseMatrix::setILUValAt(int i, int j, double val) {
  
  int rowIndex = row[i];
  int nextRowIndex = row[i + 1];
  int dist = nextRowIndex - rowIndex;
  double *data = valILU + rowIndex;
  int *dataCol = col + rowIndex;
  
  
  if (*dataCol > j) return;
  if (*dataCol == j) {
    *data = val;
    return;
  }
  
  for(int m = 1; m < dist; m++) {
    if(dataCol[m] == j) {
      data[m] = val;
      return;
    }
  }
  return;
}



double SparseMatrix::getValAt(int i, int j) {
  
  int rowIndex = row[i];
  int nextRowIndex = row[i + 1];
  int dist = nextRowIndex - rowIndex;
  double *data = val + rowIndex;
  int *dataCol = col + rowIndex;
  
  
  if (*dataCol > j) return 0;
  if (*dataCol == j) return *data;
  
  for(int m = 1; m < dist; m++) {
    if(dataCol[m] == j) return data[m];
  }
  return 0;
}


double SparseMatrix::getILUValAt(int i, int j) {
  
  int rowIndex = row[i];
  int nextRowIndex = row[i + 1];
  int dist = nextRowIndex - rowIndex;
  double *data = valILU + rowIndex;
  int *dataCol = col + rowIndex;
  
  
  if (*dataCol > j) return 0;
  if (*dataCol == j) return *data;
  
  for(int m = 1; m < dist; m++) {
    if(dataCol[m] == j) return data[m];
  }
  return 0;
}


void SparseMatrix::saveToFile() {
  
  std::ostringstream buildfilename;
  buildfilename << "./res/m_Blocksize_";
  buildfilename << nBlock;
  buildfilename << ".txt";
  std::string filename = buildfilename.str(); 
  
  std::ofstream myfile;
  myfile.open(filename.c_str());
  
  if (myfile.is_open()) {
  
    //save single values
    myfile << n << "\n";
    myfile << nBlock << "\n";
    myfile << valSize << "\n";
    myfile << rowSize << "\n";
  
    //save Arrays
    for(int i = 0; i < valSize; i++) {
      myfile << val[i] << "\n";
    }

    for(int i = 0; i < valSize; i++) {
      myfile << valILU[i] << "\n";
    }
    
    for(int i = 0; i < valSize; i++) {
      myfile << col[i] << "\n";
    }

    for(int i = 0; i < rowSize; i++) {
      myfile << row[i] << "\n";
    }
  } else {
    std::cout << "file Error" << "\n";
  }
  myfile.close();
}

void SparseMatrix::calculateEntrys(int nBlock, std::vector<ent*> *list) {
  
  // diag
  for(int k = 0; k < nBlock * nBlock; k++) {
    
    ent *newE = (ent*) malloc(sizeof(struct entry));
    newE->i = k;
    newE->j = k;
    newE->val = 4;
    (*list).push_back(newE); 
  }
  
  // sidediag
  for(int k = 0; k < nBlock * nBlock; k++) {
    // over diag
    if(k % nBlock != nBlock - 1) {
    
      ent *newE = (ent*) malloc(sizeof(struct entry));
      newE->i = k;
      newE->j = k + 1;
      newE->val = -1;
      (*list).push_back(newE);
    }
    
    // under diag
    if(k % nBlock != 0) {
    
      ent *newE = (ent*) malloc(sizeof(struct entry));
      newE->i = k;
      newE->j = k - 1;
      newE->val = -1;
      (*list).push_back(newE);
    } 
  }
  
  // side Blocks
  for(int k = 0; k < nBlock * (nBlock - 1); k++) {
  
    ent *newE = (ent*) malloc(sizeof(struct entry));
    newE->i = k;
    newE->j = k + nBlock;
    newE->val = -1;
    (*list).push_back(newE);
  }
  
  for(int k = 0; k < nBlock * (nBlock - 1); k++) {
  
    ent *newE = (ent*) malloc(sizeof(struct entry));
    newE->i = k + nBlock;
    newE->j = k;
    newE->val = -1;
    list->push_back(newE);
  }
  
  
  std::sort(list->begin(), list->end(), SparseMatrix::PComp);
}

SparseMatrix::SparseMatrix(std::string filename) {
  
  std::ifstream myfile;
  myfile.open(filename.c_str());
  
  if(myfile.is_open()) {
    
    // load single values
    myfile >> this->n;
    myfile >> this->nBlock;
    myfile >> this->valSize;
    myfile >> this->rowSize;
    
    val = new double[valSize];
    valILU = new double[valSize];
    col = new int[valSize];
    row = new int[rowSize];
    
    
    // load arrays
    for(int i = 0; i < valSize; i++) {
      myfile >> val[i];
    }

    for(int i = 0; i < valSize; i++) {
      myfile >> valILU[i];
    }
    
    for(int i = 0; i < valSize; i++) {
      myfile >> col[i];
    }

    for(int i = 0; i < rowSize; i++) {
      myfile >> row[i];
    }
  
  } else {
    std::cout << "file Error" << "\n";
  } 
  myfile.close();
}



SparseMatrix::SparseMatrix(int nBlock) {
  this->n = nBlock*nBlock;
  this->nBlock = nBlock;
  
  std::vector<ent*> list;
  calculateEntrys(nBlock, &list);
  this->valSize = list.size();
  this->rowSize = n + 1;
  
  val = new double[valSize];
  valILU = new double[valSize];
  col = new int[valSize];
  row = new int[rowSize];
  
  row[0] = 0;
  row[n] = valSize;
  int lastRowEntry = 0;
  int rowCount = 0;
  
  for(int k = 0; k < valSize; k++) {
    
    val[k] = list.at(k)->val;
    valILU[k] = val[k];
    col[k] = list.at(k)->j;
    
    if(list.at(k)->i != lastRowEntry) {
      rowCount++;
      row[rowCount] = k;
      lastRowEntry = list.at(k)->i; 
    }
    free(list.at(k));
  }
}

SparseMatrix::~SparseMatrix() {
  delete[] val;
  delete[] valILU;
  delete[] col;
  delete[] row;
}