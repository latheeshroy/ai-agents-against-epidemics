#include "utils.hpp"
#include <iostream>
#include <algorithm>
#include <cmath>


namespace utils{
    
    
    /// @brief return an index for a joint action/obs
    /// @param indiv_indices: individual actions/obs
    /// @param max_indices: number of allowed action/obs by agent, /!\ required: > 0 for all agents 
    /// @param agents_number: number of agents
    /// @return the index, e.g. to construct rewards[x][u_index]
    /// @example vector<int> a{1,0,0},b{2,1,2}; int index = get_jointIndex(a, b, 3);
    int get_jointIndex(const vector<int>& indiv_indices, const vector<int>& max_indices, int agents_number){
        // check inputs 
        assert(indiv_indices.size() == (size_t)agents_number);
        assert(max_indices.size() == (size_t)agents_number);
        assert(all_of(max_indices.begin(), max_indices.end(), [](int number){return number > 0;}));

        // compute index
        int index = indiv_indices[agents_number - 1];
        for (int i = agents_number - 2, product = 1; i >= 0; i--){
            product *= max_indices[i + 1];
            index += indiv_indices[i] * product;        
        }
        return index;
    }

    /// @brief Compute the inverse of get_jointIndex.
    /// Fill indiv_indices.
    /// @example vector<int> a{6,0,3},b{7,5,4}; int jointIndex = get_jointIndex(a, b, 3); get_indivIndices(a, b, 3, jointIndex);
    vector<int> get_indivIndices(const vector<int>& max_indices, int agents_number, int jointIndex){
        vector<int> indiv_indices(agents_number);
        for (int agent = 0; agent < agents_number; agent ++){
            int prod = accumulate(max_indices.begin() + agent + 1, max_indices.end(), 1, std::multiplies<int>());
            int q = jointIndex / prod;
            indiv_indices[agent] = q;
            jointIndex = jointIndex % prod;
        }
        return indiv_indices;
    }

    /////////////////////////////
    // ANSI 
    // https://en.wikipedia.org/wiki/ANSI_escape_code
    /////////////////////////////

    string _get_colorStr(int color){
        return "\033[" + std::to_string(color) + "m"; 
    }

    string _get_colorStr(initializer_list<int> color_list){
        string msg;
        for (int color: color_list)
            msg += _get_colorStr(color);  
        return  msg; 
    }

    void _print_color(int color, string msg){
        cout << _get_colorStr(color) << msg << _get_colorStr(Colors::reset);
    }

    void _print_color(initializer_list<int> color_list, string msg){
        cout << _get_colorStr(color_list) << msg << _get_colorStr(Colors::reset);
    }

    string get_solver_repr(string solver_name){
        return _get_colorStr(Colors::cyan)  
             + "===== " + solver_name + " =====" 
             + _get_colorStr(Colors::reset);
    }

    string print_problem(string filename, int horizon, int agents_number){
        return _get_colorStr({Colors::magenta, Colors::underline})
                + "======== PROBLEM: " + filename 
                + " horizon " + std::to_string(horizon) 
                + " agents " + std::to_string(agents_number) 
                + " ========" 
                + _get_colorStr(Colors::reset);
    }

    void print_red(string msg){
        _print_color(Colors::red, msg);
    }

    void print_green(string msg){
        _print_color(Colors::green, msg);
    }

    void print_yellow(string msg){
        _print_color(Colors::yellow, msg);
    }

    void print_magenta(string msg){
        _print_color(Colors::magenta, msg);
    }

    void print_cyan(string msg){
        _print_color(Colors::cyan, msg);
    }

    /////////////////////////////
    // seed
    /////////////////////////////

    void seed_init(int seed, bool verbose){
        if (seed == -1){ // random seed
            MARKTIME
            int wasting_time = utils::fast_calc::fastRandRange(10);
            seed = TIME * 1e9 + wasting_time;
        }
        utils::fast_calc::seed_init(seed);
        if (verbose){
            cout << "=== seed init === " << seed << endl;
        }
    }

    /////////////////////////////
    // logger
    /////////////////////////////

    ///@brief Constructor: open file.
    Logger::Logger(string filename, Verbose verbose, bool append_mode)
    : filename(filename), verbose(verbose), append_mode(append_mode){
        if (!this->filename.empty()){
            if (this->verbose > utils::Verbose::low){
                cout << "Logger open file" << this->filename << endl;
                cout << "append_mode " << this->append_mode << endl;
            }
            if (!this->append_mode)
                this->file.open(filename, ios::out | ios::trunc); // erase file content
            else
                this->file.open(filename, ios::out | ios::app); // will write at the end
        }
    }

    ///@brief Destructor: close file.
    Logger::~Logger(){
        if (!this->filename.empty()){
            if (this->verbose > utils::Verbose::low)
                cout << "Logger close file" << this->filename << endl;
            this->file.close();
        }
    }

    /// @brief Write in file (write the list of string as a csv row, i.e. with "," separator).
    /// @param list A list of string.
    void Logger::write(const initializer_list<string> & list, double time){
        this->last_write_time = time;
        //-- print ?
        if (this->verbose){
            int width = 150 / (double)list.size();
            for (auto it = list.begin(); it != list.end(); it++){
                cout.width(width);
                cout << *it;
            }
            cout << endl; 
        }
        //-- write in file ?
        if (!this->filename.empty()){
            for (auto it = list.begin(); it < list.end() - 1; it++)
                this->file << *it << ",";
            this->file << *(list.end() - 1) << endl; 
        }
    }

    double Logger::get_time_sinceLastWrite(double current_time){
        return current_time - this->last_write_time;
    }


}