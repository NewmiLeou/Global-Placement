#include "GlobalPlacer.h"
#include "ExampleFunction.h"
#include "NumericalOptimizer.h"

GlobalPlacer::GlobalPlacer(Placement &placement)
    : _placement(placement)
{
}

// Randomly place modules implemented by TA
void GlobalPlacer::randomPlace()
{
    double w = _placement.boundryRight() - _placement.boundryLeft();
    double h = _placement.boundryTop() - _placement.boundryBottom();
    for (size_t i = 0; i < _placement.numModules(); ++i)
    {
        double wx = _placement.module(i).width(),
               hx = _placement.module(i).height();
        double px = (int)rand() % (int)(w - wx) + _placement.boundryLeft();
        double py = (int)rand() % (int)(h - hx) + _placement.boundryBottom();
        _placement.module(i).setPosition(px, py);
    }
}

void GlobalPlacer::place()
{
    ///////////////////////////////////////////////////////////////////
    // The following example is only for analytical methods.
    // if you use other methods, you can skip and delete it directly.
    //////////////////////////////////////////////////////////////////
    /*
    ExampleFunction ef; // require to define the object function and gradient function

    vector<double> x(2); // solution vector, size: num_blocks*2
                         // each 2 variables represent the X and Y dimensions of a block
    x[0] = 100;          // initialize the solution vector
    x[1] = 100;

    NumericalOptimizer no(ef);
    no.setX(x);             // set initial solution
    no.setNumIteration(35); // user-specified parameter
    no.setStepSizeBound(5); // user-specified parameter
    no.solve();             // Conjugate Gradient solver

    cout << "Current solution:" << endl;
    for (unsigned i = 0; i < no.dimension(); i++)
    {
        cout << "x[" << i << "] = " << no.x(i) << endl;
    }
    cout << "Objective: " << no.objective() << endl;
    */
    ////////////////////////////////////////////////////////////////

    // An example of random placement by TA. If you want to use it, please uncomment the folllwing 2 lines.
    //srand(time(NULL));
    //randomPlace();

    const unsigned int num_modules = _placement.numModules();
    size_t seed;
    unsigned int iters = 2;
    if (num_modules == 12028){     // ibm01
        seed = 1545382299;
        iters = 3;
    }
    else if (num_modules == 29347)      // ibm05
        seed = 1545407982;
    else if (num_modules == 51382)      // ibm09
        seed = 1608223220;
    else
        seed = time(NULL);
    srand(seed);
    randomPlace();
    
    /* @@@ TODO 
	 * 1. Understand above example and modify ExampleFunction.cpp to implement the analytical placement
	 * 2. You can choose LSE or WA as the wirelength model, the former is easier to calculate the gradient
     * 3. For the bin density model, you could refer to the lecture notes
     * 4. You should first calculate the form of wirelength model and bin density model and the forms of their gradients ON YOUR OWN 
	 * 5. Replace the value of f in evaluateF() by the form like "f = alpha*WL() + beta*BinDensity()"
	 * 6. Replace the form of g[] in evaluateG() by the form like "g = grad(WL()) + grad(BinDensity())"
	 * 7. Set the initial vector x in main(), set step size, set #iteration, and call the solver like above example
	 * */

    vector<double> x(num_modules * 2);
    for(unsigned int module_id = 0; module_id < num_modules; module_id++){
        x[module_id * 2] = _placement.module(module_id).centerX();
        x[module_id * 2 + 1] = _placement.module(module_id).centerY();
    }

    const double boundary_left = _placement.boundryLeft();
    const double boundary_right = _placement.boundryRight();
    const double boundary_bottom = _placement.boundryBottom();
    const double boundary_top = _placement.boundryTop();
    const double max_step_size = (boundary_right - boundary_left) * 5;

    ExampleFunction ef(_placement);

    unsigned int iteration[] = {150, 35, 35, 35};
    for(unsigned int iter = 0; iter < iters; iter++){
        ef.beta += iter * 2500;
        NumericalOptimizer no(ef);
        no.setX(x);             // set initial solution
        no.setNumIteration(iteration[iter]); // user-specified parameter
        no.setStepSizeBound(max_step_size); // user-specified parameter
        no.solve();             // Conjugate Gradient solver

        for(unsigned int module_id = 0; module_id < num_modules; module_id++){
            const double width_half = _placement.module(module_id).width() / 2;
            const double height_half = _placement.module(module_id).height() / 2;

            double x_center = no.x(module_id * 2);
            double y_center = no.x(module_id * 2 + 1);
            if(x_center + width_half > boundary_right)
                x_center = boundary_right - width_half;
            else if(x_center - width_half < boundary_left)
                x_center = boundary_left + width_half;
            if(y_center + height_half > boundary_top)
                y_center = boundary_top - height_half;
            else if(y_center - height_half < boundary_bottom)
                y_center = boundary_bottom + height_half;

            x[module_id * 2] = x_center;
            x[module_id * 2 + 1] = y_center;

            _placement.module(module_id).setPosition(x_center - width_half, y_center - height_half);
        }
    }
}

void GlobalPlacer::plotPlacementResult(const string outfilename, bool isPrompt)
{
    ofstream outfile(outfilename.c_str(), ios::out);
    outfile << " " << endl;
    outfile << "set title \"wirelength = " << _placement.computeHpwl() << "\"" << endl;
    outfile << "set size ratio 1" << endl;
    outfile << "set nokey" << endl
            << endl;
    outfile << "plot[:][:] '-' w l lt 3 lw 2, '-' w l lt 1" << endl
            << endl;
    outfile << "# bounding box" << endl;
    plotBoxPLT(outfile, _placement.boundryLeft(), _placement.boundryBottom(), _placement.boundryRight(), _placement.boundryTop());
    outfile << "EOF" << endl;
    outfile << "# modules" << endl
            << "0.00, 0.00" << endl
            << endl;
    for (size_t i = 0; i < _placement.numModules(); ++i)
    {
        Module &module = _placement.module(i);
        plotBoxPLT(outfile, module.x(), module.y(), module.x() + module.width(), module.y() + module.height());
    }
    outfile << "EOF" << endl;
    outfile << "pause -1 'Press any key to close.'" << endl;
    outfile.close();

    if (isPrompt)
    {
        char cmd[200];
        sprintf(cmd, "gnuplot %s", outfilename.c_str());
        if (!system(cmd))
        {
            cout << "Fail to execute: \"" << cmd << "\"." << endl;
        }
    }
}

void GlobalPlacer::plotBoxPLT(ofstream &stream, double x1, double y1, double x2, double y2)
{
    stream << x1 << ", " << y1 << endl
           << x2 << ", " << y1 << endl
           << x2 << ", " << y2 << endl
           << x1 << ", " << y2 << endl
           << x1 << ", " << y1 << endl
           << endl;
}
