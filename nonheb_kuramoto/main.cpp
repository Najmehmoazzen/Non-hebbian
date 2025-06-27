// C++ Program to demonstrate Mathematical model (kuramoto single layer)
#include"Kuramoto.Version4.h"//library Kuramoto version 4 (ubuntu version push in github)

int main() {
    // Hint1: count_rows_cols_file: para in address of file that is ./data.txt
    // Hint2: read_data: first para is number of rows in data file and second para is boolean[0=dont show data,1=show data]
    // data[0]=N & data[1]=L & data[2]=a
    // data[3]=t_0 & data[4]=∆t & data[5]=t_f
    // data[6]=k_0 & data[7]=∆k & data[8]=k_f
    // data[9]=τ_0 & data[10]=∆τ & data[11]=τ_f
    double* data=read_data(count_rows_cols_file("data.txt"),0);
    double* frequency_layer1 = read_initial_1D("W=Natural frequency/natural", int(data[0]));
    double* Phases_initial_layer1 = read_initial_1D("P=Initial Phases/theta", int(data[0]));//Initial Phases  P
    int** adj_layer1 = read_initial_2D("A=Intralayer adjacency matrix/Layer1", int(data[0]));//adjacency matrix  A
    double Delay_variable = data[9];
    // while (Delay_variable < (data[11])){Delay_variable+=data[10]} // Delay loop
    double* Phases_next_layer1 = new double[int(data[0])];
    double** Phases_history_delay_layer1 = memory_of_delay_of_phases(int(data[0]),Delay_variable,data[4],Phases_initial_layer1);
    double* Phases_layer1_previous = shift_pi2_phases(int(data[0]),Delay_variable,data[4], Phases_history_delay_layer1);//Phases changer
    // Hint3: When i change it that add variable to data.txt
    ofstream Avg_Sync(name_file_data("Save/Avg_Sync/",data,12)+".txt");
    double Coupling_variable = data[6];
    cout<<"8. G to Coupling_variable. :)"<<endl;
    while (Coupling_variable <= (data[8])) { // Coupling loop
        //time_t start_calculate_time = time(NULL);
        //filesystem::create_directories(name_file_data("Save/Phases/layer1/",data,12));
        ofstream Save_phases_for_each_coupling("Save/Phases/k="+to_string(Coupling_variable)+".txt");
        ofstream Save_r_vs_t("./Save/r_vs_t/r_vs_t_k="+to_string(Coupling_variable)+".txt");

        double Total_synchrony_layer1 = 0;
        int counter_of_total_sync =0;
        double Time_variable = data[3];// reset time for new time
        while (Time_variable < (data[5] + data[4])) {
            Connected_Constant_Runge_Kutta_4(data,Delay_variable, Coupling_variable, frequency_layer1, adj_layer1, Phases_layer1_previous, Phases_history_delay_layer1,Phases_next_layer1);
            double synchrony_layer1 = order_parameter(int(data[0]), Phases_layer1_previous);// order parameters
            Save_r_vs_t<<Time_variable<< '\t'  << synchrony_layer1<< endl;

            
            Save_phases_for_each_coupling << Time_variable << '\t';
            for (int i = 0; i < int(data[0]); i++) {
                Save_phases_for_each_coupling << Phases_layer1_previous[i] << '\t';
            }
            Save_phases_for_each_coupling << endl;
            if (Time_variable >= int(data[5] * 0.8)) {// add sync to total sync
                Total_synchrony_layer1 += synchrony_layer1;
                counter_of_total_sync+=1;
            }

            // start non-hebbian function
            if (Time_variable >= int(data[5] * 0.5) && Time_variable <= int(data[5] * 0.9)) {
                non_heb(int(data[0]), adj_layer1,Phases_layer1_previous);
            }
            Time_variable += data[4];
        }
        Total_synchrony_layer1=Total_synchrony_layer1/counter_of_total_sync;// calculate total sync and pint it
        //time_t end_calculate_time = time(NULL);// end of calculate time
        //cout<< Coupling_variable << '\t' << Total_synchrony_layer1 <<'\t' <<"Execution Time: "<< (double)(end_calculate_time-start_calculate_time)<<" Seconds"<<endl;
        //Avg_Sync << Coupling_variable << '\t' << Total_synchrony_layer1<< '\t' << (double)(end_calculate_time-start_calculate_time) << endl;
        Avg_Sync << Coupling_variable << '\t' << Total_synchrony_layer1<< endl;
        Save_phases_for_each_coupling.close();
        Coupling_variable += data[7];// next Coupling_variable

        Save_r_vs_t.close();

    }
    

    // Save final adjacency matrix after non-Hebbian rewiring (after all coupling steps)
    ofstream final_adj("Save/Adj_new/adj_new.txt");
    for (int i = 0; i < int(data[0]); i++) {
        for (int j = 0; j < int(data[0]); j++) {
            final_adj << adj_layer1[i][j];
            if (j < int(data[0]) - 1) final_adj << '\t';
        }
        final_adj << endl;
    }
    final_adj.close();
    
    
    write_last_phase("Save/Last_Phase/",data,12,Phases_layer1_previous);
    Avg_Sync.close();
    delete Phases_layer1_previous;
    delete Phases_next_layer1;
    delete Phases_history_delay_layer1[0];
    return 0;
}
