#include "oSarsa.hpp"
#include "../core/distribution.hpp"
#include <condition_variable>
#include <deque>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>


namespace decPOMDP{
namespace sequential{
using namespace std;


oSarsa::oSarsa():
    horizon_tau(SEARCH.horizon * PROBLEM.agents_number),
    oState0(PROBLEM.belief_init),
    policy(SEARCH.horizon, PROBLEM.agents_number),
    final_policy(policy),
    Qfunctions(horizon_tau),
    reachable_sets(horizon_tau),
    Qversions(horizon_tau, 0),
    logger(SEARCH.log_filename, SEARCH.verbose){

    if (!SEARCH.log_filename.empty()){
        this->phase_log.open(SEARCH.log_filename + ".phases.csv");
        this->phase_log << "wall_event,search_time\n";
        this->phase_log << "constructor_start,0\n" << flush;
    }

    //-- solve the MDP (for heuristic selection of decision-rules)
    if (this->phase_log.is_open())
        this->phase_log << "mdp_start,0\n" << flush;
    this->mdp_sol.solve();
    if (this->phase_log.is_open())
        this->phase_log << "mdp_done,0\n" << flush;

    //-- blind actions (for heuristic selection of decision-rules)
    auto [blind_ju, temp] = Decision_Rule::blind_heuristic();
    this->blind_actions = utils::get_indivIndices(PROBLEM.actions_number_byAgent, PROBLEM.agents_number, blind_ju);

    //-- load multiple initial beliefs from file (if provided)
    this->_load_beliefs();
    if (this->phase_log.is_open())
        this->phase_log << "constructor_done,0\n" << flush;
}


/// @brief main public function
void oSarsa::solve(){

    //-- verbose (solver info)
    if (SEARCH.verbose){
        cout << utils::get_solver_repr("decPOMDP:oSarsa:solve() ") << endl;
    }

    //-- initialize the algo
    this->logger.write({"time", "iter", "value", "estim", "best", "Q sizes", "epsilon"});
    if (!SEARCH.log_filename.empty()){
        this->async_log.open(SEARCH.log_filename + ".async.csv");
        this->async_log << "elapsed_time,worker,stage,update_id,downstream_version_used,"
                        << "downstream_latest_version,staleness,q_updates,total_updates,"
                        << "started_at,finished_at\n";
        this->metrics_log.open(SEARCH.log_filename + ".metrics.csv");
        this->metrics_log << "elapsed_time,trajectories,total_updates,updates_per_second,"
                          << "trajectories_per_second,mean_staleness,max_staleness\n";
        if (!this->phase_log.is_open()){
            this->phase_log.open(SEARCH.log_filename + ".phases.csv");
            this->phase_log << "wall_event,search_time\n";
        }
        this->phase_log << "solve_start,0\n" << flush;
        if (SEARCH.workers > 1){
            this->execution_trace.open(SEARCH.log_filename + ".trace");
        }
    }
    if (!SEARCH.lazy_reachability){
        if (this->phase_log.is_open())
            this->phase_log << "reachability_start,0\n" << flush;
        this->_set_reachables();
        if (this->phase_log.is_open())
            this->phase_log << "reachability_done,0\n" << flush;
    }
    SEARCH.start();
    if (this->phase_log.is_open())
        this->phase_log << "policy_init_start," << SEARCH.get_time() << "\n" << flush;
    this->_init_policy();
    if (this->phase_log.is_open())
        this->phase_log << "policy_init_done," << SEARCH.get_time() << "\n" << flush;
    if (SEARCH.lazy_reachability){
        this->_invalidate_lazy_Q();
        if (this->phase_log.is_open())
            this->phase_log << "initial_q_start," << SEARCH.get_time() << "\n" << flush;
        this->_ensure_initial_Q();
        if (this->phase_log.is_open())
            this->phase_log << "initial_q_done," << SEARCH.get_time() << "\n" << flush;
    } else if (SEARCH.workers == 1) {
        this->_update_Q();
    } else {
        this->_update_Q_parallel();
    }
    
    if (SEARCH.verbose >= 10){ // debug
        cout << "POLICY init " << endl << this->policy << endl;
        cout << "Q_VALUES init" << endl;
        for (int tau = 0; tau < horizon_tau; tau++){
            auto [step, agent] = Occupancy_State::get_step_agent(tau);
            cout << "=== step " << step << " agent " << agent << endl;
            cout << this->Qfunctions.at(tau).get_repr(step, agent);
        }
    }
    
    //-- main loop
    int iter = 0;
    this->_log(iter);
    double log_period = SEARCH.timeout / 250.0;
    while(!SEARCH.is_timeout()){
        iter++;
        this->epsilon = this->_get_epsilon(iter);
        this->temperature = this->_get_temperature(this->epsilon);
        this->_update_policy();
        if (SEARCH.lazy_reachability){
            this->_ensure_initial_Q();
        } else if (SEARCH.workers == 1) {
            this->_update_Q();
        } else {
            this->_update_Q_parallel();
        }
        this->_advance_belief_idx(); // cycle to next initial belief
        
        //-- log
        double time_sinceLastLog = this->logger.get_time_sinceLastWrite(SEARCH.get_time());
        if (iter == 1 || this->improved || time_sinceLastLog > log_period)
            this->_log(iter);
    }

    this->_log(iter);
    if (!SEARCH.log_filename.empty()){
        ofstream policy_file(SEARCH.log_filename + ".policy");
        policy_file << this->final_policy;
        ofstream staleness_file(SEARCH.log_filename + ".staleness.csv");
        staleness_file << "staleness,count\n";
        for (const auto & [staleness, count]: this->staleness_histogram)
            staleness_file << staleness << ',' << count << '\n';
    }

    //-- verbose (final policy)
    if (SEARCH.verbose >= 10){
        cout << "POLICY final " << endl << this->final_policy << endl;
    }
}


/// @brief Build initial policy: \\
/// evaluate 1) the blind strategy, 2) the MDP heuristic \\
/// and keep the best. Cycles through all initial beliefs.
void oSarsa::_init_policy(){
    Policy new_policy(SEARCH.horizon, PROBLEM.agents_number);
    int n_beliefs = this->oStates0.size();
    
    for (int i = 0; i < 2; i++){
        if (i > 0 && !SEARCH.use_portfolio)
            break;
        // Try each initial belief and keep the policy that gives the best average
        double R_sum_total = 0;
        for (int b = 0; b < n_beliefs; b++){
            auto oState = this->oStates0[b];
            double R_sum = 0;
            Policy bp_policy(SEARCH.horizon, PROBLEM.agents_number);
            for (int tau = 0; tau < horizon_tau; tau++){
                auto [step, agent] = Occupancy_State::get_step_agent(tau);
                Decision_Rule & dr = bp_policy.get_decisionRule(step, agent);

                //-- select the decision rule
                if (i == 0)
                    dr = this->_select_blind(oState);
                else
                    dr = this->_select_MDP(oState);

                dr.default_action = this->blind_actions.at(agent);
                this->_extend_jointDecisionRule_LPE(dr, oState);

                //-- update R_sum, oState
                double discount = pow(PROBLEM.discount, step);
                R_sum += discount * this->rew_shaping.get_expected_reward(oState, dr);
                if (tau < this->horizon_tau - 1){
                    oState.is_compressed = false;
                    oState = oState.do_step(dr);
                    if (SEARCH.compress)
                        oState.compress();
                }
            }
            R_sum_total += R_sum;
            // Merge decision rules from this belief into new_policy
            for (int tau = 0; tau < horizon_tau; tau++){
                auto [step, agent] = Occupancy_State::get_step_agent(tau);
                for (auto & ih_iu: bp_policy.get_decisionRule(step, agent).iHist_action_map){
                    new_policy.get_decisionRule(step, agent).set_action(ih_iu.first, ih_iu.second);
                }
            }
        }
        double avg_R = R_sum_total / n_beliefs;
        if (avg_R > this->final_eval){
            this->final_eval = avg_R;
            this->policy = new_policy;
        }
    }

    this->improved_lastTau = horizon_tau - 1;
    this->final_policy = this->policy;
}


/// @brief For each time-step, select epsilon-greedy decision rules. \\
/// If cumulative rewards along the trajectory is better than current_eval, then update this->policy. \\
/// We also follow simulated annealing method. \\
/// Uses round-robin cycling through initial beliefs (multi-belief mode).
void oSarsa::_update_policy(){
    Policy new_policy(SEARCH.horizon, PROBLEM.agents_number);
    double R_sum = 0; // sum of expected rewards along the trajectory
    this->improved_lastTau = -1;
    
    auto & oState0_current = this->_get_current_oState0();
    if (SEARCH.lazy_reachability)
        this->_ensure_Q(oState0_current, this->policy);
    double best_eval = this->_get_Qvalue(oState0_current, this->policy.get_decisionRule(0, 0));
    
    auto oState = oState0_current;
    for (int tau = 0; tau < horizon_tau; tau++){
        auto [step, agent] = Occupancy_State::get_step_agent(tau);
        double discount = pow(PROBLEM.discount, step);
        if (SEARCH.lazy_reachability)
            this->_ensure_Q(oState, this->policy);
        
        //-- select the decision rule
        Decision_Rule dr = this->_select_epsilon_greedy(oState);
        
        //-- update R_sum, new_policy, oState
        R_sum += discount * this->rew_shaping.get_expected_reward(oState, dr);
        new_policy.get_decisionRule(step, agent) = dr;
        if (tau < this->horizon_tau - 1){
            oState.is_compressed = false; // in order to maintain uncompressed support (and not just one-step uncompressed)
            oState = oState.do_step(dr);
            if (SEARCH.compress)
                oState.compress();
        }

        //-- evaluate
        double new_eval = R_sum;
        if (tau < horizon_tau - 1){
            auto [next_step, next_agent] = Occupancy_State::get_step_agent(tau + 1);
            auto & dr_next = this->policy.get_decisionRule(next_step, next_agent);
            if (SEARCH.lazy_reachability)
                this->_ensure_Q(oState, this->policy);
            new_eval += discount * PROBLEM.discount * this->_get_Qvalue(oState, dr_next);
        }

        //-- improved (or "simulated annealing" criterion) ?
        double diff_eval = new_eval - best_eval;
        if (diff_eval > 1e-6){
            best_eval = new_eval;
            this->improved_lastTau = tau;
        }
        else if (this->_accept_decrease(-diff_eval, agent) && SEARCH.use_simulatedAnnealing){
            this->improved_lastTau = tau;
        }
    }
    //-- policy improved ?
    if (this->improved_lastTau >= 0){
        //-- copy the new_policy plan, i.e. actions for all <x,o> met in the new_trajectory
        //--    but keep policy actions for <x,o> met on previous trajectories that are not met in this trajectory
        for (int tau = 0; tau <= this->improved_lastTau; tau++){
            auto [step, agent] = Occupancy_State::get_step_agent(tau);
            for (auto & ih_iu: new_policy.get_decisionRule(step, agent).iHist_action_map){
                this->policy.get_decisionRule(step, agent).set_action(ih_iu.first, ih_iu.second);
            }
        }
    }

    this->improved = best_eval > this->final_eval;
    if (this->improved){
        this->final_eval = best_eval;
        this->final_policy = this->policy;
    }
    if (SEARCH.lazy_reachability && this->improved_lastTau > 0)
        this->_invalidate_lazy_Q(this->improved_lastTau - 1);
}


/// @brief Update q_t(x,o,u^,iu), backward in time, for time-steps t in [0; Tau], \\
/// @brief for each <x,o,u^> in the reachable set, and for each individual action iu, \\
/// @brief where u^ denotes u^{:agent-1} (sequential framework) \\
/// @brief and where Tau is the last step for which policy was updated during current iteration.
void oSarsa::_update_Q(){

    for (int tau = this->improved_lastTau; tau >= 0; tau--){
        auto [step, agent] = Occupancy_State::get_step_agent(tau);
        for (Support supp: this->reachable_sets.at(tau)){ // supp = <x,o,u^>
            for (int iu = 0; iu < PROBLEM.actions_number_byAgent[agent]; iu++){
                supp.set_iAction(agent, iu); // supp = <x,o,u^,iu>
                this->_update_q(supp, this->policy, tau);
                this->q_updates++;
            }
        }
        this->Qversions.at(tau)++;
    }
}


/// @brief Compute complete stage backups on a dynamic worker queue. Workers read
/// immutable downstream snapshots; only this calling thread integrates results.
void oSarsa::_update_Q_parallel(){
    if (this->improved_lastTau < 0)
        return;

    struct Stage_Result{
        int worker;
        int tau;
        uint64_t downstream_version;
        uint64_t q_updates;
        double started_at;
        double finished_at;
        shared_ptr<Qvalues_Tabular> q_function;
    };

    deque<int> tasks;
    for (int tau = this->improved_lastTau; tau >= 0; tau--)
        tasks.push_back(tau);

    vector<shared_ptr<const Qvalues_Tabular>> snapshots(this->horizon_tau);
    for (int tau = 0; tau < this->horizon_tau; tau++)
        snapshots.at(tau) = make_shared<const Qvalues_Tabular>(this->Qfunctions.at(tau));

    mutex task_mutex;
    mutex snapshot_mutex;
    mutex result_mutex;
    mutex start_mutex;
    condition_variable result_ready;
    condition_variable start_ready;
    deque<Stage_Result> results;
    int calculator_count = min(
        max(1, SEARCH.workers - 1),
        static_cast<int>(tasks.size())
    );
    int calculators_ready = 0;
    bool start_calculation = false;

    auto calculate = [&](int worker){
        int tau;
        {
            lock_guard<mutex> lock(task_mutex);
            tau = tasks.front();
            tasks.pop_front();
        }
        {
            unique_lock<mutex> lock(start_mutex);
            calculators_ready++;
            start_ready.notify_all();
            start_ready.wait(lock, [&]{ return start_calculation; });
        }

        while (true){

            shared_ptr<const Qvalues_Tabular> downstream;
            uint64_t downstream_version = 0;
            if (tau < this->horizon_tau - 1){
                lock_guard<mutex> lock(snapshot_mutex);
                downstream = snapshots.at(tau + 1);
                downstream_version = this->Qversions.at(tau + 1);
            }

            Policy policy_snapshot = this->policy;
            auto q_function = make_shared<Qvalues_Tabular>();
            uint64_t update_count = 0;
            double started_at = SEARCH.get_time();
            auto [step, agent] = Occupancy_State::get_step_agent(tau);
            for (Support supp: this->reachable_sets.at(tau)){
                for (int iu = 0; iu < PROBLEM.actions_number_byAgent[agent]; iu++){
                    supp.set_iAction(agent, iu);
                    double q_estim = this->_estimate_q(
                        supp, policy_snapshot, tau, downstream.get());
                    q_function->set_value(supp, q_estim);
                    update_count++;
                }
            }

            {
                lock_guard<mutex> lock(result_mutex);
                results.push_back({
                    worker, tau, downstream_version, update_count,
                    started_at, SEARCH.get_time(), q_function
                });
            }
            result_ready.notify_one();

            {
                lock_guard<mutex> lock(task_mutex);
                if (tasks.empty())
                    return;
                tau = tasks.front();
                tasks.pop_front();
            }
        }
    };

    vector<thread> calculators;
    calculators.reserve(calculator_count);
    for (int worker = 1; worker <= calculator_count; worker++)
        calculators.emplace_back(calculate, worker);
    {
        unique_lock<mutex> lock(start_mutex);
        start_ready.wait(lock, [&]{ return calculators_ready == calculator_count; });
        start_calculation = true;
    }
    start_ready.notify_all();

    size_t remaining = this->improved_lastTau + 1;
    while (remaining > 0){
        Stage_Result result;
        {
            unique_lock<mutex> lock(result_mutex);
            result_ready.wait(lock, [&]{ return !results.empty(); });
            result = std::move(results.front());
            results.pop_front();
        }

        uint64_t downstream_latest = result.downstream_version;
        uint64_t update_id;
        {
            lock_guard<mutex> lock(snapshot_mutex);
            if (result.tau < this->horizon_tau - 1)
                downstream_latest = this->Qversions.at(result.tau + 1);
            this->Qfunctions.at(result.tau) = *result.q_function;
            snapshots.at(result.tau) = result.q_function;
            update_id = ++this->Qversions.at(result.tau);
        }

        uint64_t staleness = downstream_latest - result.downstream_version;
        this->stage_publications++;
        this->q_updates += result.q_updates;
        if (result.tau < this->horizon_tau - 1){
            this->staleness_count++;
            this->staleness_sum += staleness;
            this->staleness_max = max(this->staleness_max, staleness);
            this->staleness_histogram[staleness]++;
        }
        bool sample_event = this->stage_publications <= 100
            || this->stage_publications % 1000 == 0;
        if (this->async_log.is_open() && sample_event){
            this->async_log << SEARCH.get_time() << ',' << result.worker << ','
                            << result.tau << ',' << update_id << ','
                            << result.downstream_version << ',' << downstream_latest << ','
                            << staleness << ',' << result.q_updates << ','
                            << this->q_updates << ',' << result.started_at << ','
                            << result.finished_at << '\n';
        }
        if (this->execution_trace.is_open() && this->stage_publications <= 200){
            this->execution_trace << "W" << result.worker
                                  << " stage " << result.tau
                                  << " update " << update_id
                                  << " interval [" << result.started_at
                                  << ',' << result.finished_at << ']';
            if (result.tau < this->horizon_tau - 1){
                this->execution_trace << " uses Q" << result.tau + 1
                                      << "^" << result.downstream_version
                                      << " latest " << downstream_latest;
            } else {
                this->execution_trace << " terminal";
            }
            this->execution_trace << '\n';
        }
        remaining--;
    }

    for (auto & calculator: calculators)
        calculator.join();
}


double oSarsa::_estimate_q_lazy(
    Support supp,
    Policy &pi,
    int tau,
    const vector<Qvalues_Tabular> &snapshot,
    vector<Support_Val_Map> &memo,
    vector<tuple<int, Support, double>> &entries) const{
    if (snapshot.at(tau).has_value(supp))
        return snapshot.at(tau).get_value(supp);
    auto found = memo.at(tau).find(supp);
    if (found != memo.at(tau).end())
        return found->second;

    int x = supp.get_hiddenState();
    int ju = supp.get_jAction();
    auto [step, agent] = Occupancy_State::get_step_agent(tau);
    double q_estim = this->rew_shaping.get_reward(agent, x, ju);
    if (tau < this->horizon_tau - 1){
        auto [next_step, next_agent] = Occupancy_State::get_step_agent(tau + 1);
        if (agent == PROBLEM.last_agent){
            for (const auto &reached: PROBLEM.get_reachables(x, ju)){
                auto next_supp = supp.do_step(reached.y, reached.joint_obs);
                int next_iu = pi.get_iAction(
                    next_step, next_agent, next_supp.get_iHistory(next_agent));
                next_supp.set_iAction(next_agent, next_iu);
                q_estim += PROBLEM.discount * reached.proba * this->_estimate_q_lazy(
                    next_supp, pi, tau + 1, snapshot, memo, entries);
            }
        } else {
            auto next_supp = supp;
            int next_iu = pi.get_iAction(
                next_step, next_agent, next_supp.get_iHistory(next_agent));
            next_supp.set_iAction(next_agent, next_iu);
            q_estim += this->_estimate_q_lazy(
                next_supp, pi, tau + 1, snapshot, memo, entries);
        }
    }

    memo.at(tau)[supp] = q_estim;
    entries.emplace_back(tau, supp, q_estim);
    return q_estim;
}


void oSarsa::_ensure_Q(Occupancy_State &oState, Policy &pi){
    int tau = oState.get_tau();
    int agent = oState.get_next_agent();
    vector<Support> roots;
    for (const auto &supp_proba: oState.get_supports_probas(oState.is_compressed)){
        Support base_supp = supp_proba.first;
        this->reachable_sets.at(tau).insert(base_supp);
        for (int iu = 0; iu < PROBLEM.actions_number_byAgent[agent]; iu++){
            Support root = base_supp;
            root.set_iAction(agent, iu);
            if (!this->Qfunctions.at(tau).has_value(root))
                roots.push_back(root);
        }
    }
    if (roots.empty())
        return;

    vector<Qvalues_Tabular> snapshot = this->Qfunctions;

    struct Lazy_Result{
        int worker;
        Support root;
        vector<tuple<int, Support, double>> entries;
        double started_at;
        double finished_at;
    };

    auto integrate = [&](Lazy_Result &result){
        std::set<int> changed_stages;
        for (const auto &[entry_tau, entry_supp, entry_value]: result.entries){
            if (!this->Qfunctions.at(entry_tau).has_value(entry_supp)){
                this->Qfunctions.at(entry_tau).set_value(entry_supp, entry_value);
                this->q_updates++;
                changed_stages.insert(entry_tau);
            }
        }
        for (int changed_tau: changed_stages)
            this->Qversions.at(changed_tau)++;

        uint64_t downstream_latest = tau < this->horizon_tau - 1
            ? this->Qversions.at(tau + 1) : 0;
        // The recursive result carries every missing dependency it used and is
        // integrated as one logical publication, so its effective lag is zero.
        uint64_t downstream_used = downstream_latest;
        uint64_t staleness = 0;
        this->stage_publications++;
        if (tau < this->horizon_tau - 1){
            this->staleness_count++;
            this->staleness_sum += staleness;
            this->staleness_max = max(this->staleness_max, staleness);
            this->staleness_histogram[staleness]++;
        }
        bool sample_event = this->stage_publications <= 100
            || this->stage_publications % 1000 == 0;
        if (this->async_log.is_open() && sample_event){
            this->async_log << SEARCH.get_time() << ',' << result.worker << ','
                            << tau << ',' << this->Qversions.at(tau) << ','
                            << downstream_used << ',' << downstream_latest << ','
                            << staleness << ',' << result.entries.size() << ','
                            << this->q_updates << ',' << result.started_at << ','
                            << result.finished_at << '\n';
        }
        if (this->execution_trace.is_open() && this->stage_publications <= 200){
            this->execution_trace << "W" << result.worker << " lazy stage " << tau
                                  << " interval [" << result.started_at << ','
                                  << result.finished_at << "] entries "
                                  << result.entries.size() << " downstream "
                                  << downstream_used << " latest "
                                  << downstream_latest << '\n';
        }
    };

    if (SEARCH.workers == 1){
        Policy policy_snapshot = pi;
        vector<Support_Val_Map> memo(this->horizon_tau);
        for (Support root: roots){
            Lazy_Result result{0, root, {}, SEARCH.get_time(), 0};
            this->_estimate_q_lazy(
                root, policy_snapshot, tau, snapshot, memo, result.entries);
            result.finished_at = SEARCH.get_time();
            integrate(result);
        }
        return;
    }

    deque<Support> tasks(roots.begin(), roots.end());
    deque<Lazy_Result> results;
    mutex task_mutex;
    mutex result_mutex;
    condition_variable result_ready;
    int calculator_count = min(
        max(1, SEARCH.workers - 1), static_cast<int>(tasks.size()));

    auto calculate = [&](int worker){
        Policy policy_snapshot = pi;
        vector<Support_Val_Map> memo(this->horizon_tau);
        while (true){
            Support root;
            {
                lock_guard<mutex> lock(task_mutex);
                if (tasks.empty())
                    return;
                root = tasks.front();
                tasks.pop_front();
            }
            Lazy_Result result{worker, root, {}, SEARCH.get_time(), 0};
            this->_estimate_q_lazy(
                root, policy_snapshot, tau, snapshot, memo, result.entries);
            result.finished_at = SEARCH.get_time();
            {
                lock_guard<mutex> lock(result_mutex);
                results.push_back(std::move(result));
            }
            result_ready.notify_one();
        }
    };

    vector<thread> calculators;
    calculators.reserve(calculator_count);
    for (int worker = 1; worker <= calculator_count; worker++)
        calculators.emplace_back(calculate, worker);

    size_t remaining = roots.size();
    while (remaining > 0){
        Lazy_Result result;
        {
            unique_lock<mutex> lock(result_mutex);
            result_ready.wait(lock, [&]{ return !results.empty(); });
            result = std::move(results.front());
            results.pop_front();
        }
        integrate(result);
        remaining--;
    }
    for (auto &calculator: calculators)
        calculator.join();
}


void oSarsa::_ensure_initial_Q(){
    for (auto &initial_state: this->oStates0)
        this->_ensure_Q(initial_state, this->policy);
}


void oSarsa::_invalidate_lazy_Q(int last_tau){
    if (last_tau < 0)
        last_tau = this->horizon_tau - 1;
    for (int tau = 0; tau <= last_tau; tau++){
        this->Qfunctions.at(tau).clear();
        this->Qversions.at(tau)++;
    }
}


/// @brief Update q(x,o,u^,iu):  \\
/// @brief if last_agent, q(x,o,ju) = r(x,ju) + discount * sum_{x',o'} Pr(x',o'|x,ju) * q_func(x',o',iu') \\
/// @brief else, q(x,o,u^,iu) = r(x,u^,iu) + q_func(x',o',<u^,iu>,iu')
/// @brief with iu' from the current policy.
/// @note r = reward shaping
/// @param supp a concise representation of <x,o,u^,iu>
void oSarsa::_update_q(Support supp, Policy &pi, int tau){
    const Qvalues_Tabular * q_func_next = tau < this->horizon_tau - 1
        ? &this->Qfunctions.at(tau + 1)
        : nullptr;
    double q_estim = this->_estimate_q(supp, pi, tau, q_func_next);

    //-- deterministic oMDP => LEARNING RATE = 1
    this->Qfunctions.at(tau).set_value(supp, q_estim);
}


double oSarsa::_estimate_q(
    Support supp,
    Policy &pi,
    int tau,
    const Qvalues_Tabular * q_func_next) const{
    int x = supp.get_hiddenState();
    int ju = supp.get_jAction(); // ju = u^{1:agent}
    auto [step, agent] = Occupancy_State::get_step_agent(tau);
    auto [next_step, next_agent] = Occupancy_State::get_step_agent(tau + 1);
    double q_estim;
    if (agent == PROBLEM.last_agent){
        q_estim = this->rew_shaping.get_reward(agent, x, ju);// PROBLEM.rewards_matrix(x, ju);
        if (step < SEARCH.horizon - 1){
            for (const auto & reached: PROBLEM.get_reachables(x, ju)){
                auto next_supp = supp.do_step(reached.y, reached.joint_obs); // <x',o'>
                int next_iu = pi.get_iAction(next_step, next_agent, next_supp.get_iHistory(next_agent)); 
                next_supp.set_iAction(next_agent, next_iu); // <x',o',u^1>
                double next_q = q_func_next->get_value(next_supp);
                q_estim += PROBLEM.discount * reached.proba * next_q;
            }
        }
    }
    else{
        auto next_supp = supp;
        int next_iu = pi.get_iAction(next_step, next_agent, next_supp.get_iHistory(next_agent)); 
        next_supp.set_iAction(next_agent, next_iu); // <x,o,<u^,iu>,iu'>
        q_estim = this->rew_shaping.get_reward(agent, x, ju) + q_func_next->get_value(next_supp);
    }
    return q_estim;
}


Decision_Rule oSarsa::_select_epsilon_greedy(Occupancy_State & oState){
    //-- pick the selection mode
    if (oState.get_next_agent() == 0){
        this->greedy = this->epsilon < utils::fast_calc::rand_double(0, 1);
        if (SEARCH.use_portfolio)
            this->rnd_0_1 = utils::fast_calc::rand_double(0, 1);    
    }
    else if (utils::fast_calc::rand_range(PROBLEM.agents_number) == 0){
        if (SEARCH.use_portfolio)
            this->rnd_0_1 = utils::fast_calc::rand_double(0, 1);    
    }

    //-- select a decision rule (greedy or an heuristic)
    Decision_Rule dr;
    if (this->greedy){
        dr = this->_select_greedy(oState);
    }
    else{
        if (this->rnd_0_1 < .25)
            dr = this->_select_blind(oState);
        else if (this->rnd_0_1 < .5)
            dr = this->_select_MDP(oState);
        else
            dr = this->_select_random(oState); 
    }
    this->_extend_jointDecisionRule_LPE(dr, oState);
    return dr;
}


/// @brief Compute the greedy decision rule for oState s w.r.t. the Q values.
Decision_Rule oSarsa::_select_greedy(Occupancy_State & oState){
    Decision_Rule greedy_dr;
    int agent = oState.get_next_agent();
    auto const & q_func = this->Qfunctions.at(oState.get_tau());
    //-- for each individual history ih we are looking the greedy iu w.r.t. Q
    for (const auto & ih_supportsProba: oState.get_private_oState(agent, oState.is_compressed)){
        double best_val = -INFINITY;
        for (int iu = 0; iu < PROBLEM.actions_number_byAgent[agent]; iu++){
            double val = 0;
            for (const auto & supp_proba: ih_supportsProba.second){
                auto supp = supp_proba.first;
                supp.set_iAction(agent, iu);
                val += supp_proba.second * q_func.get_value(supp);
            }
            if (val > best_val){
                best_val = val;
                greedy_dr.set_action(ih_supportsProba.first, iu);
            }
        }
    }
    return greedy_dr; 
}


/// @brief Compute a random decision-rule associated to an oState, \\
/// @brief i.e. for each individual history of the next agent i, a sample of a uniform distribution on U^i.
Decision_Rule oSarsa::_select_random(Occupancy_State & oState){
    Decision_Rule dr;

    int agent = oState.get_next_agent();
    int actions_number = PROBLEM.actions_number_byAgent.at(agent);
    for (const Support & ih: oState.get_iHistories(agent, oState.is_compressed)){
        int random_iu = utils::fast_calc::fastRandRange(actions_number);
        dr.set_action(ih, random_iu);
    }

    return dr;
}


/// @brief Select the greedy decision rule w.r.t. the MDP solution: \\
/// for each individual history ih, select the greedy individual action w.r.t. the MDP.
Decision_Rule oSarsa::_select_MDP(Occupancy_State & oState){
    Decision_Rule dr;
    int agent = oState.get_next_agent();
    int step = oState.get_step();
    //-- for each ih we select the greedy iu w.r.t. MDP
    for (const auto & ih_supportsProba: oState.get_private_oState(agent, oState.is_compressed)){
        double best_val = -INFINITY;
        for (int iu = 0; iu < PROBLEM.actions_number_byAgent[agent]; iu++){
            double val = 0;
            for (const auto & supp_proba: ih_supportsProba.second){
                auto supp = supp_proba.first;
                supp.set_iAction(agent, iu);
                val += supp_proba.second * mdp_sol.get_value(supp, agent + 1, step);
            }
            if (val > best_val){
                best_val = val;
                dr.set_action(ih_supportsProba.first, iu);
            }
        }
    }
    return dr;
}


/// @brief Compute the blind decision-rule,
/// i.e. for each individual history, the  individual action is the blind action.
Decision_Rule oSarsa::_select_blind(Occupancy_State & oState){
    Decision_Rule dr;
    int agent = oState.get_next_agent();
    for (const Support & ih: oState.get_iHistories(agent, oState.is_compressed)){
        dr.set_action(ih, this->blind_actions.at(agent));
    }
    return dr;
}


double oSarsa::_get_epsilon(int iter) const{
    if (iter == 1)
        return 0; // we want a greedy search after init
    
    //-- epsilon is linear w.r.t. time, from 1 to 0
    return (1.0 - SEARCH.get_time() / SEARCH.timeout) *  SEARCH.epsilon_start;
}


/// @brief "simulated annealing" criterion (to avoid local optimum): \\
/// acceptance rule of non-improving strategy.
bool oSarsa::_accept_decrease(double decrease, int agent) const{
    if (this->temperature < 1e-5)
        return false;
    if (decrease < -1e-6)
        return false;
    if (agent < PROBLEM.last_agent)
        return false;
    if (utils::fast_calc::fastRandRange(SEARCH.horizon) > 0)
        return false;
    double proba_accept = exp(-decrease / this->temperature);
    return utils::fast_calc::rand() < proba_accept;
}


double oSarsa::_get_temperature(double epsilon) const{
    return 4 * epsilon;
}


/// @brief Compute (exact) value: compute the oState trajectory, then value = the cumulative expective rewards. \\
/// @brief From tau0 (the tau of oState) to horizon.
/// @note Should not be useful (only for debug) as the Q value estimate is exact.
double oSarsa::_get_exactValue(Occupancy_State oState, Policy & pi) const{
    double R_sum = 0; // sum of expected rewards along the trajectory
    double discount = PROBLEM.discount;
    int tau = oState.get_tau();
    while (tau < this->horizon_tau){
        auto & dr = pi.get_decisionRule(oState.get_step(), oState.get_next_agent());
        R_sum += discount * oState.get_expected_reward(dr);
        if (tau < this->horizon_tau - 1){
            oState.is_compressed = false; // in order to maintain uncompressed support (and not just one-step uncompressed)
            oState = oState.do_step(dr);
            if (SEARCH.compress)
                oState.compress();
            discount *= PROBLEM.discount;
        }
        tau++;
    }
    return R_sum;
}


/// @brief Compute q(s,a) = sum_{<x,o,u^>} s(<x,o,u^>) * q(<x,o,iu>), \\
/// @brief where iu = dr(o) is the individual action and u^ denotes u^{:agent-1} (sequential framework). \\
/// @brief Elements of oState support are supposed to contain u^{:agent-1}.
/// @param dr: the decision rule to evaluate.
double oSarsa::_get_Qvalue(Occupancy_State & oState, Decision_Rule & dr) const{
    double value = 0;
    int tau = oState.get_tau();
    int agent = oState.get_next_agent();
    const auto & q_func = this->Qfunctions.at(tau);
    for (const auto & supp_proba: oState.get_supports_probas(oState.is_compressed)){
        auto supp = supp_proba.first; // <x,o,u^>
        int iu = dr.get_action(supp.get_iHistory(agent));
        supp.set_iAction(agent, iu); // <x,o,u^,iu>
        double q_supp = q_func.get_value(supp);
        value += supp_proba.second * q_supp;
    }
    return value;
}


/// @brief Compute sets <x,o,u^> which are reachables from all initial beliefs,
/// for each time-step and each agent.
/// In multi-belief mode, the reachable set is the UNION across all initial beliefs.
void oSarsa::_set_reachables(){
    //-- for tau = 0
    //--    reachables: {<x,o,u^>} == {<x>, belief[x] > 0 for any initial belief}
    for (const auto & oS0 : this->oStates0){
        for (const auto & supp_proba : oS0.get_supports_probas(false)){
            int x = supp_proba.first.get_hiddenState();
            if (supp_proba.second > 0){
                Support supp;
                supp.set_hiddenState(x);
                this->reachable_sets.at(0).insert(supp);
            }
        }
    }
    // Also include states from PROBLEM.belief_init (as fallback)
    for (int x = 0; x < PROBLEM.states_number; x++){
        if (PROBLEM.belief_init[x] > 0){
            Support supp;
            supp.set_hiddenState(x);
            this->reachable_sets.at(0).insert(supp);
        }
    }

    //-- for tau > 0
    for (int tau = 1; tau < horizon_tau; tau++){
        auto [step, agent] = Occupancy_State::get_step_agent(tau);
        if (agent == 0){
            //-- reachables: union of {<x',o'>} reached from <x,o>, for each <x,o> in the set of agent 0 at step - 1 
            for (Support supp: this->reachable_sets.at(tau - 1)){
                int x = supp.get_hiddenState();
                for (auto x_z: PROBLEM.reachables_from_x.at(x)){
                    Support supp_next = supp.do_step(x_z.first, x_z.second);
                    this->reachable_sets.at(tau).insert(supp_next);
                }
            }
        }
        else{
            //-- reachables: {<x,o,u^,iu>, iu \in U^{agent-1}, <x,o,u^> in reachables[agent-1]}
            for (Support supp: this->reachable_sets.at(tau - 1)){
                for (int iu = 0; iu < PROBLEM.actions_number_byAgent[agent - 1]; iu++){
                    supp.set_iAction(agent - 1, iu);
                    this->reachable_sets.at(tau).insert(supp);
                }
            }
        }
    }

    //-- verbose
    if (SEARCH.verbose > utils::Verbose::medium){
        cout << "Reachables <x,o>, i.e. <hidden state, joint history>:" << endl;
        for (int tau = 0; tau < horizon_tau; tau++){
            auto [step, agent] = Occupancy_State::get_step_agent(tau);
            cout << "\t step " << step 
                << " agent " << agent 
                << " size " << this->reachable_sets.at(tau).size()
                << endl;
        }
    }
}

/// @brief If LPE compression is active, we extend the policy to the whole uncompressed support. \\
/// @brief Compressed policy is built on history labels. Here we copy their action, to other equivalent joint histories. \\
/// @brief See [Optimally solving Dec-POMDPs as continuous-state MDPs: Theory and algorithms, 2014, Def. 19] for details on labels.
void oSarsa::_extend_jointDecisionRule_LPE(Decision_Rule & dr, Occupancy_State & oState) const{
    if (!oState.is_compressed)
        return;

    int agent = oState.get_next_agent();
    for (const auto & supp_proba: oState.get_supports_probas(false)){
        Support supp = supp_proba.first;
        Support label = oState.get_label(supp);
        Support ih = supp.get_iHistory(agent);
        Support ih_label = label.get_iHistory(agent);
        if (ih == ih_label)
            continue;
        int action = dr.get_action(ih_label);
        dr.set_action(ih, action);
    }
}


/// @brief Print some information about the algorithm computations.
/// @brief In multi-belief mode, value is the AVERAGE across all initial beliefs,
/// @brief and per-belief values are printed in the Q_sizes column (after iter 1).
/// @param iter iteration number 
void oSarsa::_log(int iter){
    //-- prepare data to log
    ostringstream Q_sizes;
    if (iter <= 1){
        for (int tau = 0; tau < horizon_tau; tau++){
            auto [step, agent] = Occupancy_State::get_step_agent(tau);
            if (agent == 0 && tau > 0)
                Q_sizes << "|";
            Q_sizes << this->Qfunctions.at(tau).size() << " ";
        }            
    }
    
    //-- compute value: average over all initial beliefs
    int n_beliefs = this->oStates0.size();
    double value_sum = 0;
    double eval_sum = 0;
    ostringstream per_belief_values;
    for (int b = 0; b < n_beliefs; b++){
        double v = this->_get_exactValue(this->oStates0[b], this->policy);
        double e = this->_get_Qvalue(this->oStates0[b], this->policy.get_decisionRule(0, 0));
        value_sum += v;
        eval_sum += e;
        if (b > 0) per_belief_values << ";";
        per_belief_values << v;
    }
    double value = value_sum / n_beliefs;
    double eval = eval_sum / n_beliefs;
    
    //-- log: value is the average, Q_sizes contains per-belief values after iter 1
    string extra = Q_sizes.str();
    if (iter > 1 && n_beliefs > 1){
        extra = per_belief_values.str();
    }
    
    double elapsed_time = SEARCH.get_time();
    this->logger.write({
                        to_string(elapsed_time),
                        to_string(iter),
                        to_string(value),
                        to_string(eval),
                        to_string(this->final_eval),
                        extra,
                        to_string(this->epsilon)
                        }, elapsed_time);
    if (this->metrics_log.is_open()){
        double mean_staleness = this->staleness_count == 0
            ? 0.0
            : static_cast<double>(this->staleness_sum) / this->staleness_count;
        double updates_per_second = elapsed_time > 0 ? this->q_updates / elapsed_time : 0.0;
        double trajectories_per_second = elapsed_time > 0 ? iter / elapsed_time : 0.0;
        this->metrics_log << elapsed_time << ',' << iter << ',' << this->q_updates << ','
                          << updates_per_second << ',' << trajectories_per_second << ','
                          << mean_staleness << ',' << this->staleness_max << '\n';
    }
}


/// @brief Load multiple initial beliefs from a file.
/// @brief File format: one belief per line, space-separated probabilities over S' states.
/// @brief Lines starting with '#' are comments. Empty lines are skipped.
/// @brief If no file is provided, uses the single PROBLEM.belief_init.
void oSarsa::_load_beliefs(){
    if (SEARCH.beliefs_file.empty()){
        // Single-belief mode: use the default oState0
        this->oStates0.push_back(this->oState0);
        if (SEARCH.verbose >= utils::Verbose::low)
            cout << "  [multi-belief] Single belief mode (no --beliefs-file)" << endl;
        return;
    }

    ifstream file(SEARCH.beliefs_file);
    if (!file.is_open()){
        cerr << "ERROR: Cannot open beliefs file: " << SEARCH.beliefs_file << endl;
        // Fallback to single belief
        this->oStates0.push_back(this->oState0);
        return;
    }

    int nS = PROBLEM.states_number;
    string line;
    int count = 0;
    while (getline(file, line)){
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#')
            continue;
        
        istringstream iss(line);
        Vector belief(nS);
        double val;
        int idx = 0;
        while (iss >> val && idx < nS){
            belief[idx] = val;
            idx++;
        }
        
        if (idx != nS){
            cerr << "WARNING: belief line " << count + 1 
                 << " has " << idx << " values, expected " << nS 
                 << ". Skipping." << endl;
            continue;
        }

        // Normalize (safety)
        double sum = 0;
        for (int i = 0; i < nS; i++) sum += belief[i];
        if (abs(sum - 1.0) > 1e-6){
            for (int i = 0; i < nS; i++) belief[i] /= sum;
        }

        this->oStates0.emplace_back(belief);
        count++;
    }
    file.close();

    if (this->oStates0.empty()){
        cerr << "WARNING: No valid beliefs in file. Using default." << endl;
        this->oStates0.push_back(this->oState0);
    }

    if (SEARCH.verbose >= utils::Verbose::low){
        cout << "  [multi-belief] Loaded " << this->oStates0.size() 
             << " initial beliefs from " << SEARCH.beliefs_file << endl;
    }
}


/// @brief Get the current initial oState for round-robin cycling.
Occupancy_State & oSarsa::_get_current_oState0(){
    return this->oStates0[this->current_belief_idx];
}


/// @brief Advance to the next initial belief (round-robin).
void oSarsa::_advance_belief_idx(){
    this->current_belief_idx = (this->current_belief_idx + 1) % this->oStates0.size();
}


}}