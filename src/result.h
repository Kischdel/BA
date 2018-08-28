#ifndef RESULT_H
#define RESULT_H

typedef struct resultFGMRES {
  int restart, iter, steps, restarts;
  double time, timeJacobi, averageTimeJacobi, timeFgmres, averageTimeFgmres;
} resultFGMRES;


#endif //RESULT_H