#pragma once

#include "../core/_module.hpp"

#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <iostream>

namespace parser_light{
    using namespace std;

    struct Parser{
        //-- header data
        double discount;
        int agents_number;
        int states_number = 0;  
        vector<int> actions_number_byAgent;
        int actions_joint_number = -1;
        vector<int> observations_number_byAgent;
        int observations_joint_number = -1;
        vector<double> belief_init;
        //-- dynamics data
        unordered_map<int, unordered_map<int, unordered_map<int, double>>> dynamics_T; // dynamics_T[u][x][y]: p(y |u, x)
        unordered_map<int, unordered_map<int, unordered_map<int, double>>> dynamics_O; // dynamics_O[u][y][z]: p(z |u,y)
        algebra::Matrix<double> rewards_matrix; // rewards_matrix(x,u)

        //-- constructor
        Parser(const std::string filename);
        
        void encode_problem();
        ifstream _open_file(std::string filename);
        void _parse_header(std::string filename);
        void _parse_dynamics(std::string filename);
    };


} // end namespace