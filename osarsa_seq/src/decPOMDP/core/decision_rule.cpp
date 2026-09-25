#include "decision_rule.hpp"

namespace decPOMDP{

Decision_Rule::Decision_Rule(){}
Decision_Rule::Decision_Rule(RuleType type): type(type){}        
Decision_Rule::Decision_Rule(int blind_action): type(RuleType::BLIND_DR), blind_action(blind_action){}

int Decision_Rule::get_action(const Support & iHist, int agent){
    int action;
    auto got = this->iHist_action_map.find(iHist);

    switch (this->type){
        case RuleType::REGULAR:
            if (got != this->iHist_action_map.end()){
                action = got->second;
            }
            else{
                action = this->default_action;
            }
            break;
        case RuleType::RANDOM:
            if (got != this->iHist_action_map.end()){
                action = got->second;
            }
            else{
                action = utils::fast_calc::rand_range(PROBLEM.actions_number_byAgent[agent]);
                this->iHist_action_map[iHist] = action;
            }
            break;
        case RuleType::BLIND_DR:
            action = this->blind_action; 
            break;
        default:
            assert(false);                        
    }
    return action;
}

void Decision_Rule::set_action(const Support & iHist, int action){
    this->iHist_action_map[iHist] = action;
}

/// @brief Blind heuristic (do not use state nor observation): \\
/// @brief argmax_{actions} min_{states} reward(state, action) 
/// @return <the action, a lower bound for the reward>
pair<int, double> Decision_Rule::blind_heuristic(){
    double lower_bound = -INFINITY;
    int heuristic_action;
    for (int action = 0; action < PROBLEM.actions_joint_number; action++){            
        double reward = PROBLEM.rewards_matrix.min(0, action);
        if (reward > lower_bound){
            lower_bound = reward;
            heuristic_action = action;
        }
    }

    return make_pair(heuristic_action, lower_bound);
};


/// @brief Myopic heuristic (use state + previous agents actions): \\
/// @brief argmax_{u^{agent}} max_{u^{agent+1:}} reward(state, <u^{:agent-1},u^agent,u^{agent+1:}>) \\
/// @brief i.e. the greedy individual action w.r.t reward(state, u) given u^{:agent-1} and assuming next agents will be greedy.
/// @param supp: must contain x and u^{:agent-1}. Can contain history values (we will extract x and u).
/*pair<int, double> Decision_Rule::myopic_heuristic(int agent, Support supp){
    double lower_bound = -INFINITY;
    int heuristic_iu;
    for (int iu = 0; iu < PROBLEM.actions_number_byAgent.at(agent); iu++){
        supp.set_iAction(agent, iu);
        double reward;
        if (agent == PROBLEM.last_agent){
            int ju = supp.get_jAction();
            reward = PROBLEM.rewards_matrix(supp.get_hiddenState(), ju);
cout << "\t ju " << ju << " r " << reward << endl;
        }
        else{
            tie(std::ignore, reward) = Decision_Rule::myopic_heuristic(agent + 1, supp);
        }
        if (reward > lower_bound){
            lower_bound = reward;
            heuristic_iu = iu;
        }
    }
    return make_pair(heuristic_iu, lower_bound);
}*/


///@brief Worst heuristic: 
/// argmin_{actions} min_{states} reward(state, action)
///@return <the action, a lower bound for the reward>
pair<int, double> Decision_Rule::worst_heuristic(){
    double lower_bound = INFINITY;
    int heuristic_action;
    for (int action = 0; action < PROBLEM.actions_joint_number; action++){    
        double reward = PROBLEM.rewards_matrix.min(0, action);
        if (reward < lower_bound){
            lower_bound = reward;
            heuristic_action = action;
        }
    }
    return make_pair(heuristic_action, lower_bound);
}

///@brief Lucky heuristic: 
/// argmax_{actions} max_{states} reward(state, action)
///@return <the action, a higher bound for the reward>
pair<int, double> Decision_Rule::lucky_heuristic(){
    double higher_bound = -INFINITY;
    int heuristic_action;
    for (int action = 0; action < PROBLEM.actions_joint_number; action++){    
        double reward = PROBLEM.rewards_matrix.max(0, action);
        if (reward > higher_bound){
            higher_bound = reward;
            heuristic_action = action;
        }
    }
    return make_pair(heuristic_action, higher_bound);
}

size_t Decision_Rule::size() const{
    return this->iHist_action_map.size();
}

bool Decision_Rule::operator==(const Decision_Rule & other) const{
    return this->iHist_action_map == other.iHist_action_map;
}

string Decision_Rule::get_repr(int agent, int step) const{
    ostringstream os;
    if (this->default_action != ACTION_NOT_FOUND)
        os << "default_action " << this->default_action << endl;
    for (const auto & iHist_action: this->iHist_action_map){
        os << " <agent" << agent << ">";
        os << "  hist: " ;
        for (int oldness = min(SEARCH.truncation, step) - 1; oldness >= 0; oldness--){
            int z = iHist_action.first.get_observation(agent, oldness);
            os << " " << z;
        }
        os << " action: " <<  iHist_action.second << endl;
    }
    return os.str();
}

ostream &operator<<(ostream &os, const Decision_Rule &d){
    for (auto & iHist_action: d.iHist_action_map){
        os << iHist_action.first << " => " <<  iHist_action.second << endl;
    }
    return os;
}

//------------------------------------
//-- Joint_DecisionRule
//------------------------------------
Joint_DecisionRule::Joint_DecisionRule(){};
Joint_DecisionRule::Joint_DecisionRule(int nb_agents): decision_rules(nb_agents){}
Joint_DecisionRule::Joint_DecisionRule(const vector<Decision_Rule> & drs): decision_rules(drs){}

int Joint_DecisionRule::get_iAction(int agent, const Support & ih){
    return this->decision_rules.at(agent).get_action(ih);
}

void Joint_DecisionRule::set_iAction(int agent, const Support & ih, int action){
    this->decision_rules.at(agent).set_action(ih, action);
}

/// @brief Return a joint action.
/// @param supp a joint_history (can contain extra values: we will extract individual histories)
int Joint_DecisionRule::get_jAction(const Support & supp){
    Support temp;
    for (int agent: PROBLEM.agents){
        int iAction = this->get_iAction(agent, supp.get_iHistory(agent));
        if (iAction == ACTION_NOT_FOUND)
            return ACTION_NOT_FOUND;
        temp.set_iAction(agent, iAction);
    }
    return temp.get_jAction();
}

void Joint_DecisionRule::set_jAction(const Support & supp, int action){
    Support temp;
    temp.set_jAction(action);
    for (int agent: PROBLEM.agents){
        int iu = temp.get_iAction(agent);
        this->set_iAction(agent, supp.get_iHistory(agent), iu);
    }
}

void Joint_DecisionRule::set_default_actions(vector<int> actions){
    for (int agent: PROBLEM.agents){
        this->decision_rules.at(agent).default_action = actions.at(agent);
    }
}


ostream &operator<<(ostream &os, const Joint_DecisionRule & jdr){
    int nb_agents = jdr.decision_rules.size();
    for (int agent = 0; agent < nb_agents; agent++)
        os << "** agent " << agent << endl << jdr.decision_rules.at(agent);
    return os;
}

string Joint_DecisionRule::get_repr(int step) const{
    ostringstream os;
    int nb_agents = this->decision_rules.size();
    for (int agent = 0; agent < nb_agents; agent++){
        if (agent > 0)
            os << "\t -------------- \n";
        os <<  this->decision_rules.at(agent).get_repr(agent, step);
    }
    return os.str();
}


//------------------------------------
//-- policy
//------------------------------------

Policy::Policy(int horizon, int nb_agents): joint_drs(horizon, Joint_DecisionRule(nb_agents)){}

int Policy::get_iAction(int step, int agent, const Support & ih){
    return this->joint_drs.at(step).get_iAction(agent, ih);
}

/// @brief Return a joint action.
/// @param supp a joint_history (can contain x or u values: we will extract individual histories)
int Policy::get_jAction(int step, const Support & supp){
    return this->joint_drs.at(step).get_jAction(supp);
}

Decision_Rule & Policy::get_decisionRule(int step, int agent){
    return this->joint_drs.at(step).decision_rules.at(agent);
}

Joint_DecisionRule & Policy::get_joint_decisionRule(int step){
    return this->joint_drs.at(step);
}


void Policy::set_decisionRule(int step, int agent, const Decision_Rule & dr){
    this->joint_drs.at(step).decision_rules.at(agent) = dr;
}

ostream &operator<<(ostream &os, const Policy & pi){
    int horizon = pi.joint_drs.size();
    for (int step = 0; step < horizon; step++){
        const auto & jdr = pi.joint_drs.at(step);
        os << "==== Step " << step << endl
           << jdr.get_repr(step);
    }
    return os;
}

///////////////////////////
// Variations_Generator
//////////////////////////

vector<Decision_Rule> Variations_Generator::get_decision_rules(const unordered_set<Support> & iHistories, int actions_number){
    // get variations
    int variations_length = iHistories.size();
    const Variations & variations = this->_get_variations(variations_length, actions_number);
    int variations_number = variations.size();

    // generate decision rules
    vector<Decision_Rule> decision_rules(variations_number);
    for (int i = 0; i < variations_number; i++){
        Decision_Rule & dr = decision_rules.at(i);
        int j = 0;
        for (const Support & iHist: iHistories){
            dr.set_action(iHist, variations.at(i).at(j++));
        }
    }
    return decision_rules;
}

void Variations_Generator::clear(){
    this->variations_map.clear();
}

Variations& Variations_Generator::_get_variations(int length, int actions_number){
    //-- already generated ?
    if (this->STORE){
        auto got = this->variations_map.find(length);
        if (got != this->variations_map.end()){
            auto got2 = got->second.find(actions_number);
            if (got2 != got->second.end()){
                return got2->second;
            }
        }
    }
    // generate 
    int n_variations = pow<int, int>(actions_number, length); 
    assert(0 <= n_variations && n_variations < (int)1e8);
    vector<int> vec(length, ACTION_NOT_FOUND);
    this->variations = Variations(n_variations, vec);            
    this->length = length;
    this->actions_number = actions_number;
    this->num_variation = 0;
    this->_generate(0, vec);
    if (this->STORE){
        this->variations_map[length][actions_number] = this->variations;
    }
    return this->variations;//this->variations_map.at(length).at(actions_number);
}

void Variations_Generator::_generate(int depth, const vector<int> &variation){
    if (depth == this->length){
        variations.at(num_variation++) = variation;
        return;
    }
    for (int action = 0; action < actions_number; action++){
        vector<int> variation2 = variation;        
        variation2.at(depth) = action;
        this->_generate(depth + 1, variation2);                
    }
}

///////////////////////////
// Decision_Rules_Enumeration
//////////////////////////

Decision_Rules_Enumeration::Decision_Rules_Enumeration(int agents_number): agents_number(agents_number){
    this->decision_rules.resize(agents_number);
    this->size_byAgent.resize(agents_number, 0);
}

void Decision_Rules_Enumeration::add_agent(const unordered_set<Support> & supports, int actions_number){
    int & agent = this->agents_num;
    this->decision_rules.at(agent) = VARIATIONS_GENERATOR.get_decision_rules(supports, actions_number);
    this->size_byAgent.at(agent) = this->decision_rules.at(agent).size();
    this->size = utils::get_product(this->size_byAgent);
    agent++;
}


vector<Decision_Rule> Decision_Rules_Enumeration::get(int index){
    vector<int> indiv_indices = utils::get_indivIndices(this->size_byAgent, PROBLEM.agents_number, index);
    vector<Decision_Rule> drs(this->agents_number);
    for (int agent = 0; agent < this->agents_number; agent++){
        int dr_index = indiv_indices.at(agent);
        drs.at(agent) = this->decision_rules.at(agent).at(dr_index);
    }
    return drs;
}



}