#ifndef RESULT_H
#define RESULT_H

typedef struct resultFGMRES {
  int restart, iter, steps, restarts;
  double time, timeJacobiLower, timeJacobiUpper;
  double averageTimeJacobiLower, averageTimeJacobiUpper;
  double timeFgmres, averageTimeFgmres;
  double averageNormJacobiLower, averageNormJacobiUpper;

} resultFGMRES;


#endif //RESULT_H