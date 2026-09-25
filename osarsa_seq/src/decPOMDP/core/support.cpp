#include "support.hpp"

namespace decPOMDP{
using namespace std;
using namespace common;

int Support::N_BITS_HIDDEN_STATE; // bits number for x
int* Support::N_BITS_iOBS; // bits number for an observation by [agent]
int* Support::N_BITS_iACTIONS; // bits number for a command by [agent]

int** Support::BITS_START_iOBS; // first bit of [step][agent] observations
int* Support::BITS_START_iACTIONS; // first bit of U[agent]

uint64_t Support::MASKS[64];
uint64_t** Support::MASKS_iOBS; // MASKS_iOBS[step][agent]
uint64_t* Support::MASKS_iACTIONS; // MASKS_iACTION[agent]
uint64_t Support::MASK_ACTIONS;
uint64_t* Support::MASKS_iHISTORIES;
uint64_t Support::MASKS_jHISTORIES;
uint64_t Support::MASKS_jOBS; // the last observations (all individual last observation)
uint64_t Support::MASK_HIDDEN_STATE;

unordered_map<uint64_t, int> Support::MAP_SUPPORTS_jACTIONS;  // map: Support iACTIONS -> joint actions
uint64_t* Support::jACTIONS;  // map: joint actions -> Support iACTIONS
uint64_t* Support::jOBSERVATIONS;

///////////////////////////////////////
/// bitset representation
///////////////////////////////////////

/// @brief precompute utilities for bitset representation of supports
/// BITSET REPRESENTATION:
///  hidden state: x
///      x
///  individual control values
///      u[agent=0]
///      ...
///      u[agent=n-1]
///  historics:
///      z[step=0][agent=0], z[step=1][agent=0], ... z[step=TRUNC-1][agent=0]
///      ...
///      z[step=0][agent=n-1], z[step=1][agent=n-1], ... z[step=TRUNC-1][agent=n-1]
///  => total: bits_required
void Support::init(){
    if (SEARCH.verbose >= utils::Verbose::medium)
        cout << "=== Support::init()" << endl;
    int start = 0;

    //-- MASKS
    MASKS[0] = 0UL;
    for (int i = 1; i < 64; i++)
        MASKS[i] = MASKS[i-1] | (1UL << (i - 1));

    //-- hidden state x
    N_BITS_HIDDEN_STATE = ceil(log2<int>(PROBLEM.states_number));
    MASK_HIDDEN_STATE = MASKS[N_BITS_HIDDEN_STATE];
    start += N_BITS_HIDDEN_STATE;

    //-- individual actions
    N_BITS_iACTIONS = new int[PROBLEM.agents_number];
    BITS_START_iACTIONS = new int[PROBLEM.agents_number];
    MASKS_iACTIONS = new uint64_t[PROBLEM.agents_number];
    MASK_ACTIONS = 0UL;
    for (int agent: PROBLEM.agents){
        BITS_START_iACTIONS[agent] = start;
        N_BITS_iACTIONS[agent] = ceil(log2<int>(PROBLEM.actions_number_byAgent[agent])); 
        MASKS_iACTIONS[agent] = MASKS[N_BITS_iACTIONS[agent]] << start;
        MASK_ACTIONS |= MASKS_iACTIONS[agent];
        start += N_BITS_iACTIONS[agent];
    }

    //-- Histories
    N_BITS_iOBS = new int[PROBLEM.agents_number];
    BITS_START_iOBS = new int*[SEARCH.truncation];
    MASKS_iOBS = new uint64_t*[SEARCH.truncation];
    MASKS_iHISTORIES = new uint64_t[PROBLEM.agents_number];
    MASKS_jHISTORIES = 0UL;
    MASKS_jOBS = 0UL;
    for (int agent: PROBLEM.agents){            
        N_BITS_iOBS[agent] = ceil(log2<int>(PROBLEM.observations_number_byAgent[agent]));
    }
    for (int step = 0; step < SEARCH.truncation; step++){
        BITS_START_iOBS[step] = new int[PROBLEM.agents_number];
        MASKS_iOBS[step] = new uint64_t[PROBLEM.agents_number];
    }
    for (int agent: PROBLEM.agents){
        MASKS_iHISTORIES[agent] = 0UL;
        for (int step = 0; step < SEARCH.truncation; step++){
            BITS_START_iOBS[step][agent] = start;
            MASKS_iOBS[step][agent] = MASKS[N_BITS_iOBS[agent]] << start;
            MASKS_iHISTORIES[agent] |= MASKS_iOBS[step][agent];
            start += N_BITS_iOBS[agent];
        }
        MASKS_jHISTORIES |= MASKS_iHISTORIES[agent];
        if (SEARCH.truncation > 0)
            MASKS_jOBS |= MASKS_iOBS[SEARCH.truncation - 1][agent];
    }
    

    //-- bits required
    int bits_required = N_BITS_HIDDEN_STATE
                        + accumulate(N_BITS_iACTIONS, N_BITS_iACTIONS + PROBLEM.agents_number, 0)
                        + SEARCH.truncation * accumulate(N_BITS_iOBS, N_BITS_iOBS + PROBLEM.agents_number, 0);
    if (SEARCH.verbose >= utils::Verbose::medium)
        cout << "bits required " << bits_required << " (truncation " << SEARCH.truncation << ")" << endl;
    assert(start == bits_required);
    assert(bits_required <= (int)sizeof(uint64_t)*8);

    //-- joint actions map
    MAP_SUPPORTS_jACTIONS.clear();
    jACTIONS = new uint64_t[PROBLEM.actions_joint_number];
    for (int jaction = 0; jaction < PROBLEM.actions_joint_number; jaction++){
        vector<int> actions = utils::get_indivIndices(PROBLEM.actions_number_byAgent, PROBLEM.agents_number, jaction);
        Support supp;
        for (int agent: PROBLEM.agents){
            supp.set_iAction(agent, actions.at(agent));
        }
        Support supp_act(supp.container & MASK_ACTIONS);
        MAP_SUPPORTS_jACTIONS[supp_act.container] = jaction;
        jACTIONS[jaction] = supp_act.container;
    }

    //-- joint obs
    jOBSERVATIONS = new uint64_t[PROBLEM.observations_joint_number];
    for (int z = 0; z < PROBLEM.observations_joint_number; z++){
        Support supp((uint64_t)0);
        vector<int> iObs = PROBLEM.observations_joint2indiv[z];
        for (int agent: PROBLEM.agents){
            supp.set_iObservation(agent, iObs[agent]);
        }
        jOBSERVATIONS[z] = supp.get_container();
    }

}

///////////////////////////////////////
/// constructors
///////////////////////////////////////
Support::Support(){}

Support::Support(uint64_t container): container(container){}

Support::Support(int x, Support jh, int ju){
    *this = jh;
    this->set_hiddenState(x);
    this->set_jAction(ju);
}

///////////////////////////////////////
/// get methods
///////////////////////////////////////

uint64_t Support::get_container() const{
    return container;
}

int Support::get_hiddenState() const{
    return container & MASK_HIDDEN_STATE;
}

int Support::get_iAction(int agent) const{
    return (container & MASKS_iACTIONS[agent]) >> BITS_START_iACTIONS[agent];
}

vector<int> Support::get_iActions() const{
    vector<int> actions(PROBLEM.agents_number);
    for (int agent: PROBLEM.agents)
        actions[agent] = this->get_iAction(agent);
    return actions;
}

int Support::get_jAction() const{
    return MAP_SUPPORTS_jACTIONS.at(container & MASK_ACTIONS);
}

Support Support::get_iHistory(int agent) const{            
    return container & MASKS_iHISTORIES[agent];
}


Support Support::get_jHistory() const{            
    return container & MASKS_jHISTORIES;
}

/// @brief Build a support composed of histories for agents [first_agent : PROBLEM.last_agent],
/// i.e. return the joint history where histories af agents [0: first_agent -1] has been erased.
Support Support::get_partial(int first_agent) const{    
    Support partial(this->get_jHistory());
    for (int agent = 0; agent < first_agent; agent++){
        partial.clear_iHistory(agent);
    }
    return partial.container;
}

bool Support::operator==(const Support & other) const{
    return other.container == this->container;
}

bool Support::operator!=(const Support & other) const{
    return other.container != this->container;
}

bool Support::operator<(const Support & other) const{
    return this->container < other.container;
}

Support Support::get_iHist_masked(int agent) const {
   return (container & ~MASKS_iHISTORIES[agent]); // mask the agent individual History
}


///////////////////////////////////////
/// set methods
///////////////////////////////////////

void Support::set_hiddenState(int state){
    container = (container & ~MASK_HIDDEN_STATE) | state;
}

void Support::set_iAction(int agent, int action){
    container = (container & ~MASKS_iACTIONS[agent]) | ((uint64_t)action << BITS_START_iACTIONS[agent]);
}

void Support::set_jAction(int jAction){
    container = (container & ~MASK_ACTIONS) | jACTIONS[jAction];
}

void Support::set_iObservation(int agent, int observation){
    if (SEARCH.truncation > 1){
        this->_shift_iObservations(agent);
    }
    else if (SEARCH.truncation == 0){
        return;
    }
    container &= ~MASKS_iOBS[SEARCH.truncation-1][agent]; // erase old value
    container |= ((uint64_t)observation << BITS_START_iOBS[SEARCH.truncation-1][agent]); // set new value
}

void Support::set_jObservation(int jObs){
    if (SEARCH.truncation > 1){
        for (int agent: PROBLEM.agents){
            this->_shift_iObservations(agent);
        }
    }
    container = (container & ~MASKS_jOBS) | jOBSERVATIONS[jObs];
}

void Support::clear_actions(){
    container &= ~MASK_ACTIONS;
}

void Support::clear_iHistory(int agent){
    container &= ~MASKS_iHISTORIES[agent];
}

void Support::set_iHistory(int agent, const Support& other){
   this->clear_iHistory(agent);
   container |= other.container; // set other individual History
}

Support Support::do_step(int hidden_state, int jObs) const{
    Support supp_next = *this;
    supp_next.set_hiddenState(hidden_state);
    supp_next.set_jObservation(jObs);
    supp_next.clear_actions();
    return supp_next;
}

///////////////////////////////////////
/// protected
///////////////////////////////////////

void Support::_shift_iObservations(int agent){
    uint64_t val = (container >> N_BITS_iOBS[agent]) & MASKS_iHISTORIES[agent];
    container = (container & ~MASKS_iHISTORIES[agent]) | val;
}

int Support::get_observation(int agent, int oldness) const{ 
    assert(oldness < SEARCH.truncation);
    int idx = SEARCH.truncation - oldness - 1;
    return (container & MASKS_iOBS[idx][agent]) >> BITS_START_iOBS[idx][agent];
}


///////////////////////////////////////
/// print / debug
///////////////////////////////////////
ostream &operator<<(ostream &os, const Support &supp){
    os << supp.get_repr();
    return os;
}

string Support::get_zs_repr(int step, int agent) const{
    ostringstream os;
    for (int oldness = min(SEARCH.truncation, step) - 1; oldness >= 0; oldness--){
        int z = this->get_observation(agent, oldness);
        os << " " << z;
    }
    return os.str();
}

string Support::get_ih_repr(int step, int agent) const{
    ostringstream os;
    os << " <agent" << agent << ">";
    os << "  hist: " ;
    os << get_zs_repr(step, agent);
    return os.str();
}


/// @param step for history
/// @param next_agent for action
string Support::get_repr(int step, int next_agent) const{
    ostringstream os;
    os << "x = " << this->get_hiddenState();
    for (int agent: PROBLEM.agents){
        os << " <agent" << agent << ">";
        int u = this->get_iAction(agent);
        if (agent < next_agent)
            os << " action: " <<  u;
        os << "  hist: " ;
        os << get_zs_repr(step, agent);
    }
    return os.str();
}

void Support::demo(){
    const auto randInt = utils::fast_calc::rand_range;

    cout << "=== Support demo:" << endl;

    // Randomly pick values for:
    //  - state x
    //  - step
    //  - actions / observations historics
    Support support;    
    
    int x = randInt(PROBLEM.states_number);
    int step = randInt(SEARCH.truncation);
    int last_agent = randInt(PROBLEM.agents_number);
    int u_max = PROBLEM.actions_joint_number;
    int z_max = PROBLEM.observations_joint_number;

    vector<int> seq_act;
    vector<int> seq_obs;
    for (int t=0; t <= step; t++){
        seq_act.push_back(randInt(u_max));
        seq_obs.push_back(randInt(z_max));
    }

    // Set container with these values:
    support.set_hiddenState(x);
    for (int t=0; t <= step; t++){
        vector<int> act_indivs = utils::get_indivIndices(PROBLEM.actions_number_byAgent, PROBLEM.agents_number, seq_act[t]);
        vector<int> obs_indivs = PROBLEM.observations_joint2indiv[seq_obs[t]];
        assert(PROBLEM.agents_number == (int)obs_indivs.size());
        for (int agent: PROBLEM.agents){
            support.set_iObservation(agent, obs_indivs.at(agent));
            support.set_iAction(agent, act_indivs.at(agent));
            if (t == step && agent == last_agent)
                break;
        }
        // support.set_jAction(seq_act[t]);
    }
    
    // print
    cout << "WRITE "
            << "x:" << x << " "
            << "step:" << step << " (trunc=" << SEARCH.truncation << ") " 
            << "seq_act:";
    for (int t=0; t <= step; t++)
        cout << seq_act[t] << " ";
    cout << "seq_obs:";
    for (int t=0; t <= step; t++)
        cout << seq_obs[t] << " ";
    cout << "last_agent " << last_agent;
    cout << endl;
    cout << "SUPPORT ";
    cout << support.get_repr() << endl;
}


}// end namespace