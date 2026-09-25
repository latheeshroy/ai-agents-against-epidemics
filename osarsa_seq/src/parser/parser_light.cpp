#include "parser_light.hpp"

namespace parser_light{
using namespace std;


Parser::Parser(const std::string filename){
    this->_parse_header(filename);
    this->_parse_dynamics(filename);    
}


void Parser::encode_problem(){
    using namespace common;

    //-- header
    PROBLEM.discount = this->discount;
    PROBLEM.agents_number =  this->agents_number;
    PROBLEM.states_number =  this->states_number;
    PROBLEM.belief_init = Vector(this->belief_init);
    PROBLEM.observations_number_byAgent = this->observations_number_byAgent;
    PROBLEM.actions_number_byAgent = this->actions_number_byAgent;

    PROBLEM.last_agent = PROBLEM.agents_number - 1;
    PROBLEM.observations_joint_number = this->observations_joint_number;
    PROBLEM.actions_joint_number = this->actions_joint_number;

    //-- dynamics / rewards
    PROBLEM.dynamics_T = this->dynamics_T;
    PROBLEM.dynamics_O = this->dynamics_O;
    PROBLEM.rewards_matrix = this->rewards_matrix;

    //-- finalize
    PROBLEM.finalize();
}


ifstream Parser::_open_file(std::string filename){
    ifstream infile;
    infile.open(filename);
	if(!infile){       
        string temp = "../" + filename; // maybe we're in the build folder
        infile.open(temp);
        if(!infile){
        	cout << "Could not open input file " << temp << endl; 
            throw -1;
		}
	}
    return infile;
}


/// @brief get agents number, discount and all the state, action and observation space
void Parser::_parse_header(std::string filename){
    ifstream infile = this->_open_file(filename);

    int actions_read_index = -1;
    int obs_read_index = -1;
    bool ReadActions = false;
    bool ReadObservations = false;
    bool ReadStart = false; // end at here
    
	string line;
    while (getline(infile, line) && !ReadStart){
        istringstream is(line);
		if (line.empty() || line.at(0) == '#') // empty line or comment line ?
			continue;
        string s;
        int temp_num = 0;
        bool ReadAgents = false;
        bool ReadDiscount = false;
        bool ReadStates = false;
        while (is >> s){
            if (s == "agents:"){
                ReadAgents = true;
            }
            else if (s == "discount:"){
                ReadDiscount = true;
            }
            else if (s == "states:"){
                ReadStates = true;
            }
            else if (s == "actions:"){
                ReadActions = true;
            }
            else if (s == "observations:"){
                ReadObservations = true;
            }
            else if (s == "start:"){
                ReadStart = true;
            }

            // Get Agents number， init Actions and Observations
            if (ReadAgents && temp_num == 1){
                this->agents_number = stoi(s);
                this->actions_number_byAgent = vector<int>(this->agents_number, 0);
                this->observations_number_byAgent = vector<int>(this->agents_number, 0);
            }
            // Get discount factor
            if (ReadDiscount && temp_num == 1){
                discount = stod(s);
            }
            // Get all the States
            if (ReadStates && temp_num > 0){
                this->states_number++;
            }
            // Get all actions
            if (ReadActions && actions_read_index < this->agents_number && actions_read_index > -1){
                this->actions_number_byAgent.at(actions_read_index)++;
            }
            // Get all observations
            if (ReadObservations && obs_read_index < this->agents_number && obs_read_index > -1){
                this->observations_number_byAgent.at(obs_read_index)++;
            }
            // Get intial belief
            if (ReadStart && temp_num > 0){
                double pb = stod(s);
                this->belief_init.push_back(pb);
            }
            temp_num += 1;
        }

        if (ReadActions){
            actions_read_index += 1;
        }
        if (ReadObservations){
            obs_read_index += 1;
        }
    }
    this->observations_joint_number = utils::get_product(this->observations_number_byAgent);
    this->actions_joint_number = utils::get_product(this->actions_number_byAgent);
    infile.close();
}


/// @brief get transitions, observations and rewards
void Parser::_parse_dynamics(std::string filename){
    ifstream infile = this->_open_file(filename);

    this->rewards_matrix = algebra::Matrix<double>(this->states_number, this->actions_joint_number, 0);
    string line;
    while (getline(infile, line)){
        istringstream is(line);
		if (line.empty() || line.at(0) == '#') // empty line or comment line ?
			continue;

        string s;
        int temp_num = 0;
        bool buildTrans = false;
        bool buildObs = false;
        bool buildReward = false;
        int aI = 0;
        int sI = 0;
        int oI = 0;
        int snewI = 0;
        double pb = 0;
        vector<int> actions(this->agents_number);
        vector<int> obs_indicies(this->agents_number);
        while (is >> s){
            if (s == "T:"){
                buildTrans = true;
            }
            else if (s == "O:"){
                buildObs = true;
            }
            else if (s == "R:"){
                buildReward = true;
            }

            if (0 < temp_num && temp_num < 1 + this->agents_number){
                if (buildTrans || buildObs || buildReward){
                    actions[temp_num - 1] = stoi(s);
                }
            }
            else if (temp_num == 4){
                if (buildTrans || buildObs || buildReward){
                    sI = stoi(s);
                }
            }
            else if (5 < temp_num && temp_num < 6 + this->agents_number){
                if (buildTrans){
                    if (temp_num == 6){
                        snewI = stoi(s);
                    }
                }
                if (buildObs){
                    obs_indicies[temp_num - 6] = stoi(s);
                }
            }
            else if (temp_num == 8){
                if (buildTrans){
                    pb = stod(s);
                    aI = utils::get_jointIndex(actions, this->actions_number_byAgent, this->agents_number);
                    if (pb > 1e-9)
                        this->dynamics_T[aI][sI][snewI] = pb;
                }
            }
            else if (temp_num == 9){
                if (buildObs){
                    pb = stod(s);
                    aI = utils::get_jointIndex(actions, this->actions_number_byAgent, this->agents_number);
                    oI = utils::get_jointIndex(obs_indicies, this->observations_number_byAgent, this->agents_number);
                    if (pb > 1e-9)
                        this->dynamics_O[aI][sI][oI] = pb;
                }
            }
            else if (temp_num == 10){
                if (buildReward){
                    pb = stod(s);
                    aI = utils::get_jointIndex(actions, this->actions_number_byAgent, this->agents_number);
                    this->rewards_matrix(sI,aI) = pb;
                }
            }

            temp_num += 1;
        }
    }
    infile.close();
}


} // end namespace