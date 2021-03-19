#ifndef EXAMPLEFUNCTION_H
#define EXAMPLEFUNCTION_H

#include "Placement.h"
#include "NumericalOptimizerInterface.h"

#include <cmath>

typedef struct bin {
    double x_center;
    double y_center;
} Bin;

class ExampleFunction : public NumericalOptimizerInterface
{
public:
    ExampleFunction(Placement &placement);

    void evaluateFG(const vector<double> &x, double &f, vector<double> &g);
    void evaluateF(const vector<double> &x, double &f);
    unsigned dimension();

    double beta;

private:
    Placement &_placement;
    unsigned int num_modules;
    static const unsigned int num_bins_per_row = 16;

    double Tb;  // target bin density
    Bin bins[num_bins_per_row][num_bins_per_row];
    double bin_width;
    double bin_height;
    double core_area;
    double gamma;

    double CalculateTb(const double width, const double height);
    void InitializeBins(const double width, const double height);
    double CalculateLSE(const vector<double> &x);
    double CalculateLSE(const vector<double> &x, vector<double> &g);    // for evaluateFG
    double CalculateBinDensity(const vector<double> &x);
    double CalculateBinDensity(const vector<double> &x, vector<double> &g);     // for evaluateFG
};

#endif // EXAMPLEFUNCTION_H
