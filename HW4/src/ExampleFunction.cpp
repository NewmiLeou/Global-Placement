#include "ExampleFunction.h"

ExampleFunction::ExampleFunction(Placement &placement) : _placement(placement)
{
    num_modules = _placement.numModules();

    const double width = _placement.boundryRight() - _placement.boundryLeft();
    const double height = _placement.boundryTop() - _placement.boundryBottom();
    Tb = CalculateTb(width, height);
    InitializeBins(width, height);

    gamma = (width + height) / 100;
    beta = 0;
}

//  Caculate target bin density
double ExampleFunction::CalculateTb(const double width, const double height)
{
    core_area = width * height;

    double total_area = 0;
    for (unsigned int module_id = 0; module_id < num_modules; module_id++)
        total_area += _placement.module(module_id).area();

    return total_area / core_area;
}

//  Initialize Bin
void ExampleFunction::InitializeBins(const double width, const double height)
{
    const double boundary_left = _placement.boundryLeft();
    const double boundary_bottom = _placement.boundryBottom();

    bin_width = width / num_bins_per_row;
    bin_height = height / num_bins_per_row;

    for (unsigned int y_id = 0; y_id < num_bins_per_row; y_id++) {
        double y_center = boundary_bottom + (0.5 + y_id) * bin_height;
        for (unsigned int x_id = 0; x_id < num_bins_per_row; x_id++) {
            double x_center = boundary_left + (0.5 + x_id) * bin_width;
            bins[y_id][x_id].x_center = x_center;
            bins[y_id][x_id].y_center = y_center;
        }
    }
}

//  Calculate LSE
double ExampleFunction::CalculateLSE(const vector<double> &x)
{
    double total_lse = 0;

    vector<double> x_exp_table(num_modules * 4);
    for (unsigned int i = 0; i < num_modules; i++) {
        x_exp_table[i * 4] = exp(x[i * 2] / gamma);     // e^(xi/gemma)
        x_exp_table[i * 4 + 1] = exp(-x[i * 2] / gamma);    // e^(-xi/gemma)
        x_exp_table[i * 4 + 2] = exp(x[i * 2 + 1] / gamma);     // e^(yi/gemma)
        x_exp_table[i * 4 + 3] = exp(-x[i * 2 + 1] / gamma);    // e^(-yi/gemma)
    }

    const unsigned int num_nets = _placement.numNets();
    for (unsigned int net_id = 0; net_id < num_nets; net_id++) {
        Net &net = _placement.net(net_id);

        double lse_x = 0;
        double lse_nx = 0;
        double lse_y = 0;
        double lse_ny = 0;
        const unsigned int num_pins = net.numPins();
        for (unsigned int index = 0; index < num_pins; index++) {
            const unsigned int module_id = net.pin(index).moduleId();

            lse_x += x_exp_table[module_id * 4];
            lse_nx += x_exp_table[module_id * 4 + 1];
            lse_y += x_exp_table[module_id * 4 + 2];
            lse_ny += x_exp_table[module_id * 4 + 3];
        }

        total_lse += log(lse_x) + log(lse_nx) + log(lse_y) + log(lse_ny);
    }

    return gamma * total_lse;
}

//  Calculate LSE with gradients, for evaluateFG
double ExampleFunction::CalculateLSE(const vector<double> &x, vector<double> &g)
{
    double total_lse = 0;

    vector<double> x_exp_table(num_modules * 4);
    for (unsigned int i = 0; i < num_modules; i++) {
        x_exp_table[i * 4] = exp(x[i * 2] / gamma);
        x_exp_table[i * 4 + 1] = exp(-x[i * 2] / gamma);
        x_exp_table[i * 4 + 2] = exp(x[i * 2 + 1] / gamma);
        x_exp_table[i * 4 + 3] = exp(-x[i * 2 + 1] / gamma);
    }

    const unsigned int num_nets = _placement.numNets();
    for (unsigned int net_id = 0; net_id < num_nets; net_id++) {
        Net &net = _placement.net(net_id);

        double lse_x = 0;
        double lse_nx = 0;
        double lse_y = 0;
        double lse_ny = 0;
        const unsigned int num_pins = net.numPins();
        for (unsigned int index = 0; index < num_pins; index++) {
            const unsigned int module_id = net.pin(index).moduleId();

            lse_x += x_exp_table[module_id * 4];
            lse_nx += x_exp_table[module_id * 4 + 1];
            lse_y += x_exp_table[module_id * 4 + 2];
            lse_ny += x_exp_table[module_id * 4 + 3];
        }

        total_lse += log(lse_x) + log(lse_nx) + log(lse_y) + log(lse_ny);

        for (unsigned int index = 0; index < num_pins; index++) {
            const unsigned int module_id = net.pin(index).moduleId();

            g[module_id * 2] += x_exp_table[module_id * 4] / lse_x;
            g[module_id * 2] -= x_exp_table[module_id * 4 + 1] / lse_nx;
            g[module_id * 2 + 1] += x_exp_table[module_id * 4 + 2] / lse_y;
            g[module_id * 2 + 1] -= x_exp_table[module_id * 4 + 3] / lse_ny;
        }
    }

    return gamma * total_lse;
}

//  Calculate Bin Density
double ExampleFunction::CalculateBinDensity(const vector<double> &x)
{
    double total_bin_density = 0;

    for (unsigned int y_id = 0; y_id < num_bins_per_row; y_id++) {
        for (unsigned int x_id = 0; x_id < num_bins_per_row; x_id++) {
            double bin_density = 0;
            for (unsigned int module_id = 0; module_id < num_modules; module_id++) {
                const double m_width = _placement.module(module_id).width();
                const double m_height = _placement.module(module_id).height();

                // Smoothing by Bell-Shaped Function (p.61)
                double bin_density_x = 0;
                double dx = abs(x[module_id * 2] - bins[y_id][x_id].x_center);
                if (dx <= bin_width / 2 + m_width / 2) {
                    double a = 4 / ((bin_width + m_width) * (2 * bin_width + m_width));
                    bin_density_x = 1 - a * dx * dx;
                }
                else if (dx <= bin_width + m_width / 2) {
                    double b = 4 / (bin_width * (2 * bin_width + m_width));
                    bin_density_x = b * (dx - bin_width - m_width / 2) * (dx - bin_width - m_width / 2);
                }
                else {
                    bin_density_x = 0;
                }

                double bin_density_y = 0;
                double dy = abs(x[module_id * 2 + 1] - bins[y_id][x_id].y_center);
                if (dy <= bin_height / 2 + m_height / 2) {
                    double a = 4 / ((bin_height + m_height) * (2 * bin_height + m_height));
                    bin_density_y = 1 - a * dy * dy;
                }
                else if (dy <= bin_height + m_height / 2) {
                    double b = 4 / (bin_height * (2 * bin_height + m_height));
                    bin_density_y = b * (dy - bin_height - m_height / 2) * (dy - bin_height - m_height / 2);
                }
                else {
                    bin_density_y = 0;
                }

                bin_density += bin_density_x * bin_density_y;
            }

            //double density_diff = max(bin_density - Tb, 0.0);
            //total_bin_density += (bin_density - Tb) * (bin_density - Tb);
            total_bin_density += bin_density * bin_density;
        }
    }

    return beta * total_bin_density;
}

//  Calculate Bin Density with gradients, for evaluateFG
double ExampleFunction::CalculateBinDensity(const vector<double> &x, vector<double> &g)
{
    double total_bin_density = 0;

    for (unsigned int y_id = 0; y_id < num_bins_per_row; y_id++) {
        for (unsigned int x_id = 0; x_id < num_bins_per_row; x_id++) {
            vector<double> g_temp(g.size(), 0);

            double bin_density = 0;
            for (unsigned int module_id = 0; module_id < num_modules; module_id++) {
                const double m_width = _placement.module(module_id).width();
                const double m_height = _placement.module(module_id).height();

                double bin_density_x = 0;
                double bin_density_x_g = 0;
                double dx = x[module_id * 2] - bins[y_id][x_id].x_center;
                double dx_abs = abs(dx);
                if (dx_abs <= bin_width / 2 + m_width / 2) {
                    double a = 4 / ((bin_width + m_width) * (2 * bin_width + m_width));
                    bin_density_x = 1 - a * dx_abs * dx_abs;
                    bin_density_x_g = -2 * a * dx;
                }
                else if (dx_abs <= bin_width + m_width / 2) {
                    double b = 4 / (bin_width * (2 * bin_width + m_width));
                    bin_density_x = b * (dx_abs - bin_width - m_width / 2) * (dx_abs - bin_width - m_width / 2);
                    if (dx > 0)
                        bin_density_x_g = 2 * b * (dx - bin_width - m_width / 2) * 1;
                    else
                        bin_density_x_g = 2 * b * (dx - bin_width - m_width / 2) * -1;
                }
                else {
                    bin_density_x = 0;
                    bin_density_x_g = 0;
                }

                double bin_density_y = 0;
                double bin_density_y_g = 0;
                double dy = x[module_id * 2 + 1] - bins[y_id][x_id].y_center;
                double dy_abs = abs(dy);
                if (dy_abs <= bin_height / 2 + m_height / 2) {
                    double a = 4 / ((bin_height + m_height) * (2 * bin_height + m_height));
                    bin_density_y = 1 - a * dy_abs * dy_abs;
                    bin_density_y_g = -2 * a * dy;
                }
                else if (dy_abs <= bin_height + m_height / 2) {
                    double b = 4 / (bin_height * (2 * bin_height + m_height));
                    bin_density_y = b * (dy_abs - bin_height - m_height / 2) * (dy_abs - bin_height - m_height / 2);
                    if (dy > 0)
                        bin_density_y_g = 2 * b * (dy - bin_height - m_height / 2) * 1;
                    else
                        bin_density_y_g = 2 * b * (dy - bin_height - m_height / 2) * -1;
                }
                else {
                    bin_density_y = 0;
                    bin_density_y_g = 0;
                }

                bin_density += bin_density_x * bin_density_y;

                g_temp[module_id * 2] = bin_density_x_g * bin_density_y;
                g_temp[module_id * 2 + 1] = bin_density_x * bin_density_y_g;
            }

            //double density_diff = max(bin_density - Tb, 0.0);
            //total_bin_density += (bin_density - Tb) * (bin_density - Tb);
            total_bin_density += bin_density * bin_density;

            for (unsigned int module_id = 0; module_id < num_modules; module_id++) {
                g[module_id * 2] += beta * 2 * bin_density * g_temp[module_id * 2];
                g[module_id * 2 + 1] += beta * 2 * bin_density * g_temp[module_id * 2 + 1];
            }
        }
    }

    return beta * total_bin_density;
}

void ExampleFunction::evaluateFG(const vector<double> &x, double &f, vector<double> &g)
{
    g = vector<double>(g.size(), 0);
    f = CalculateLSE(x, g);
    f += CalculateBinDensity(x, g);
}

void ExampleFunction::evaluateF(const vector<double> &x, double &f)
{
    double LSE = CalculateLSE(x);
    double Db = CalculateBinDensity(x);
    f = LSE + Db;
}

unsigned ExampleFunction::dimension()
{
    return num_modules * 2; // num_blocks*2 
    // each two dimension represent the X and Y dimensions of each block
}
