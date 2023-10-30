import torch as th


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]

def get_all_for_magail(states_or_actions):
    states_or_actions_new = states_or_actions.clone()
    return states_or_actions_new.view(states_or_actions_new.shape[0], states_or_actions_new.shape[1],
                                      states_or_actions_new.shape[2]*states_or_actions_new.shape[3])


def resolve_activate_function(name):
    if name.lower() == "relu":
        return th.nn.ReLU
    if name.lower() == "sigmoid":
        return th.nn.Sigmoid
    if name.lower() == "leakyrelu":
        return th.nn.LeakyReLU
    if name.lower() == "prelu":
        return th.nn.PReLU
    if name.lower() == "softmax":
        return th.nn.Softmax
    if name.lower() == "tanh":
        return th.nn.Tanh