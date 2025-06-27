#ifndef KURAMOTO_VERSION4_H_INCLUDED
#define KURAMOTO_VERSION4_H_INCLUDED

#include<iostream>//for cout
#include<fstream>//infile /ofstream
#include <string>//for stod( ) function= name_file_data
#include <sstream>//stringstream ss(line)
#include<ctime>//For Example clock()
#include <cmath>//For Example pow function= name_file_data
#include <omp.h>
#include <stdio.h>
#include<iomanip>//function= name_file_data
//#include <time.h>
//#include <filesystem>

using namespace std;

void Convert_next_to_history_and_previous(int Number_of_node,
                                          double Time_step,
                                          double Delay_variable,
                                          double** Phases_history_delay,
                                          double* Phases_next,
                                          double* Phases_previous)
{
    int memory = int((Delay_variable / Time_step) + 1);
    for (int i = 0; i < Number_of_node; i++) {
        for (int j = 0; j < memory - 1; j++) {
            Phases_history_delay[i][j] = Phases_history_delay[i][j + 1];
        }
        Phases_history_delay[i][memory - 1] = Phases_next[i];
    }
    for (int i = 0; i < Number_of_node; i++) { Phases_previous[i] = Phases_next[i]; }
}

// dydt
double dydt(int Number_of_phase,
            double frustration_intra_layer,
            int N,
            double dt,
            double coupling,
            double W,
            int* is_connected,
            double phi,
            double** phi_hist)
{
    double M = 0;
    double a = 0.0;
    for (int i = 0; i < N; i++){
        a += (is_connected[i] * sin((phi_hist[i][0] - phi + frustration_intra_layer)));
    }
    M = W + (coupling / (N * 1.0)) * a ;
    return M;
}

// CCRK4
void Connected_Constant_Runge_Kutta_4(double* data,
                                      double delay,
                                      double coupling,
                                      double* W,
                                      int** adj,
                                      double* y,
                                      double** Phases_history_delay,
                                      double* Phases_next)
{                                
    int Number_of_node=int(data[0]);
    for (int i = 0; i < Number_of_node; i++)
    {
        double k1 = dydt(i, data[2], Number_of_node,data[4],coupling, W[i], adj[i], y[i], Phases_history_delay);
        double k2 = dydt(i, data[2], Number_of_node,data[4],coupling, W[i], adj[i], y[i] + k1 * data[4] / 2.0,Phases_history_delay);
        double k3 = dydt(i, data[2], Number_of_node,data[4],coupling, W[i], adj[i], y[i] + k2 * data[4] / 2.0,Phases_history_delay);
        double k4 = dydt(i, data[2], Number_of_node,data[4],coupling, W[i], adj[i], y[i] + k3 * data[4], Phases_history_delay);
        Phases_next[i] = y[i] + data[4] / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
    Convert_next_to_history_and_previous(Number_of_node, data[4], delay, Phases_history_delay,Phases_next, y);
}

// order_parameter
double order_parameter(int Number_of_node, double* phi)
{
    double rc = 0.0, rs = 0.0;
    for (int j = 0; j < Number_of_node; j++)
    {
        rc += cos(phi[j]);
        rs += sin(phi[j]);
    }
    return sqrt(pow(rc, 2) + pow(rs, 2)) / (1.0 * Number_of_node);
}

// Create file name with 2 decimal places for each data element
string name_file_data(string address,double* data_text,int Number_of_row) {
    ostringstream fileName;
    fileName << address;
    int canter_i=0;
    for (canter_i = 0; canter_i < Number_of_row-1; canter_i++) {
        fileName << fixed << setprecision(2) << data_text[canter_i] << ",";
    }
    fileName << fixed << setprecision(2) << data_text[canter_i];
    //cout << "7. O Data file '"<< fileName.str() <<"'. :)" << endl;
    return fileName.str();
}

// Print last phases data
int write_last_phase(string address,double* data,int Number_of_row,double* last_Phase_layer1) {
    ofstream file_print(name_file_data(address,data,12)+".txt");
    for (int i = 0; i < int(data[0]); i++) {
        file_print << last_Phase_layer1[i] << endl;
    }
    file_print.close();
    return 0;
}

// Change the phases as the pi/2 clockwise
double* shift_pi2_phases(int Number_of_node,double Delay_variable,double Time_step, double** Phases_history_delay) {// calculate initial theta
    // number of cell to save phases in memory
    int memory = int((Delay_variable / Time_step) + 1);
    double* shifted_phase = new double[Number_of_node];// Making Array
    for (int i = 0; i < Number_of_node; i++)
    {
        if (Phases_history_delay[i][memory - 1] >= (double)(M_PI / 2.0))
        {
            shifted_phase[i] = Phases_history_delay[i][memory - 1] - (double)(1.5 * M_PI);
        }
        else
        {
            shifted_phase[i] = Phases_history_delay[i][memory - 1] + (double)(M_PI / 2.0);
        }
    }
    cout << "6. S pi2 phases. :)" << endl;
    return shifted_phase;
}

// Create history of delay of phases
double** memory_of_delay_of_phases(int Number_of_node ,double Delay_variable,double Time_step ,double* Phases_initial) {
    // number of cell to save phases in memory
    int memory = int((Delay_variable / Time_step) + 1);
    // create empty Phases memory delay
    double** Phases_memory_delay = new double* [Number_of_node];// 
    for (int i = 0; i < Number_of_node; i++) {
        Phases_memory_delay[i] = new double[memory];// [node][delay]
    }
    // first memory for initial phase
    for (int i = 0; i < Number_of_node; i++) {
        Phases_memory_delay[i][0] = Phases_initial[i];
    }
    // when i have a delay in system
    for (int t = 1; t < memory; t++)
    {
        for (int i = 0; i < Number_of_node; i++)
        {
            if (Phases_memory_delay[i][t - 1] >= (double)(M_PI / 2.0))
            {
                Phases_memory_delay[i][t] = Phases_memory_delay[i][t - 1] - (double)(1.5 * M_PI);
            }
            else
            {
                Phases_memory_delay[i][t] = Phases_memory_delay[i][t - 1] + (double)(M_PI / 2.0);
            }
        }
    }
    cout << "5. C '"<<memory<<"' cell to memory of delay of phases. :)" << endl;
    return Phases_memory_delay;
}

// Read matrix connection (2D int)
int** read_initial_2D(string Filename_address, int Number_of_node)
{
    int** data_2D = new int* [Number_of_node];
    for (int i = 0; i < Number_of_node; i++)
        data_2D[i] = new int[Number_of_node];
    ifstream file("input_data/" + Filename_address + ".txt");
    if (!file)
    {
        cout << "4. Data file Matrix is not here !? ------- error" << endl;
        return data_2D;
    }
    else
    {
        for (int i = 0; i < Number_of_node; i++)
        {
            for (int j = 0; j < Number_of_node; j++)
            {
                int elem = 0;
                file >> elem;
                data_2D[i][j] = elem;
            }
        }
    }
    cout << "4. R Data file '"<< Filename_address <<"'. :)" << endl;
    return data_2D;
}

// Read data from text 1d double
double* read_initial_1D(string Filename_address, int Number_of_node)// (Phases & frequency)
{
    double* data_1D = new double[Number_of_node];
    ifstream file("input_data/" + Filename_address + ".txt");
    for (int i = 0; i < Number_of_node; i++)
    {
        file >> data_1D[i];
    }
    file.close();
    cout << "3. R Data file '"<< Filename_address <<"'. :)" << endl;
    return data_1D;
}

// Read Data from data.txt file
// Read data from data.txt and write them to pointer 1d
double* read_data(int number_of_data,bool show)
{
    double* data = new double[number_of_data];
    string kk;
    ifstream file_data("data.txt");
    if (!file_data)
    {
        cout << "2. Data file is not here !? ------- error" << endl;
    }
    else
    {
        cout << "2. R Data file. :)" << endl;
        string line, item;
        int i = 0;
        while (i<number_of_data)
        {
            file_data >> kk;
            data[i] = stod(kk);
            i++;
        }
    }
    if (show==1)
    {
        cout << "|----------------------------------------------------------------------------------------------------|" << endl;
        cout << "|Data=>>                              \t\t\t\t\t\t\t\t     |" << endl;      //                                 ---
        cout << "|                                      \t\t\t\t\t\t\t\t     |" << endl;     //                                 ---
        cout << "|data[0]= Number =\t\t" << data[0] <<"\t\t\t\t\t\t\t\t     |" << endl;      //                                 ---
        cout << "|data[1]= Lambda =\t\t" << data[1] <<"\t\t\t\t\t\t\t\t     |" << endl;      //                                 ---
        cout << "|data[2]= Alpha =\t\t" << data[2] <<"\t\t\t\t\t\t\t\t     |" << endl;      //                                 ---
        cout << "|----------------------------------------------------------------------------------------------------|" << endl;
        cout << "|Timing=>>                              \t\t\t\t\t\t\t     |" << endl;      //                                 ---
        cout << "|                                      \t\t\t\t\t\t\t\t     |" << endl;     //                                 ---
        cout << "|data[3]= First Time =\t\t" << data[3] <<"\t\t\t\t\t\t\t\t     |" << endl;  //                                 ---
        cout << "|data[4]= dt =\t\t\t" << data[4] <<"\t\t\t\t\t\t\t\t     |" << endl;        //                                 ---
        cout << "|data[5]= Final Time =\t\t" << data[5] << "\t\t\t\t\t\t\t\t     |" << endl; //                                 ---
        cout << "|----------------------------------------------------------------------------------------------------|" << endl;
        cout << "|coupling=>>                              \t\t\t\t\t\t\t     |" << endl;    //                                 ---
        cout << "|                                      \t\t\t\t\t\t\t\t     |" << endl;     //                                 ---
        cout << "|data[6]= coupling_start =\t" << data[6] <<"\t\t\t\t\t\t\t\t     |" << endl;//                                 ---
        cout << "|data[7]= coupling_step =\t" << data[7] <<"\t\t\t\t\t\t\t\t     |" << endl; //                                 ---
        cout << "|data[8]= coupling_end =\t" << data[8] <<"\t\t\t\t\t\t\t\t     |" << endl;  //                                 ---
        cout << "|----------------------------------------------------------------------------------------------------|" << endl;
        cout << "|delay=>>                              \t\t\t\t\t\t\t\t     |" << endl;     //                                 ---
        cout << "|                                      \t\t\t\t\t\t\t\t     |" << endl;     //                                 ---
        cout << "|data[9]= delay_start =\t\t" << data[9] << "\t\t\t\t\t\t\t\t     |" << endl;//                                 ---
        cout << "|data[10]= delay_step =\t\t" << data[10] << "\t\t\t\t\t\t\t\t     |" << endl; //                                 ---
        cout << "|data[11]=delay_end =\t\t" << data[11] << "\t\t\t\t\t\t\t\t     |" << endl;//                                 ---
        cout << "|----------------------------------------------------------------------------------------------------|" << endl;
    }
    file_data.close();
    return data;
}

// count rows & columns file in data.txt and return number of rows
int count_rows_cols_file(string file1)
{
    int rows = 0, cols = 0;
    string line, item;
    ifstream file(file1);
    while (getline(file, line))
    {
        rows++;
        if (rows == 1)// First row only: 
        {
            stringstream ss(line);// Set up up a stream from this line
            while (ss >> item) cols++;// Each item delineated by spaces
        }
    }
    file.close();
    cout << "1. File had " << rows << " rows and " << cols << " columns. :)" << endl;
    return rows;
}





//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//@@@                                 Non Hebb                                       @@@@
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
/*
 * Rewires a connection in the network by removing the weakest (non-Hebbian)
 * neighbor of a randomly selected node and connecting it to a new random node
 * to which it is not already connected.
 *
 * Parameters:
 *   - N: Number of nodes in the network
 *   - adj: NxN adjacency matrix (int**)
 *   - phi: Phase values of the nodes (double[])
 */
void non_heb(int N, int** adj, double* phi) {
    // Select a random node i
    int node_i = rand() % N;

    // Find the connected neighbor with the weakest (smallest) Hebbian similarity
    int weakest_neighbor = -1;
    double min_hebb_value = 1.0;

    for (int j = 0; j < N; j++) {
        if (adj[node_i][j] == 1) {
            double hebb_value = 0.5 * (1 - cos( phi[j] - phi[node_i]));

            if (hebb_value <= min_hebb_value) {
                min_hebb_value = hebb_value;
                weakest_neighbor = j;
            }
        }
    }

    // If node_i has no neighbors, do nothing
    if (weakest_neighbor == -1) return;

    // Disconnect from the weakest neighbor
    adj[node_i][weakest_neighbor] = 0;
    adj[weakest_neighbor][node_i] = 0;

    // Try to connect to a new node that is not already connected
    int new_neighbor = -1;
    int max_attempts = 10000; // Avoid infinite loop
    int attempts = 0;

    while (attempts < max_attempts) {
        int candidate = rand() % N;
        if (candidate != node_i && adj[node_i][candidate] == 0) {
            new_neighbor = candidate;
            break;
        }
        attempts++;
    }

    // If a valid new neighbor was found, create the new connection
    if (new_neighbor != -1) {
        adj[node_i][new_neighbor] = 1;
        adj[new_neighbor][node_i] = 1;


    //cout << "Node " << node_i << " rewired from " << weakest_neighbor << " to " << new_neighbor << endl;

    }



}







#endif // KURAMOTO_VERSION4_H_INCLUDED