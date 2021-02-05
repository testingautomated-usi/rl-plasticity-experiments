import argparse

from agent import DEFAULT_N_EVAL_EPISODES, Agent
from agent_stub import AgentStub
from algo.alphatest import AlphaTest
from env_utils import instantiate_env_variables, instantiate_eval_callback
from log import Log
from utilities import check_halve_or_double, check_file_existence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo_name", choices=["ppo2", "dqn", "a2c", "sac", "td3", "ddpg", "her"], required=True, default="ppo2",
    )
    parser.add_argument("--env_name", type=str, required=True, default="CartPole-v1")
    parser.add_argument("--log_to_tensorboard", type=bool, default=False)
    parser.add_argument("--tb_log_name", type=str, default="ppo2")
    parser.add_argument("--train_total_timesteps", type=int, default=300000)
    parser.add_argument("--n_eval_episodes", type=int, default=DEFAULT_N_EVAL_EPISODES)
    parser.add_argument("--render", type=bool, default=False)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--model_to_load", type=str, default=None)
    parser.add_argument("--continue_learning", type=bool, default=False)
    parser.add_argument("--discrete_action_space", type=bool, default=False)
    parser.add_argument("--eval_callback", type=bool, default=False)
    parser.add_argument("--continue_learning_suffix", type=str, default="continue_learning")
    parser.add_argument("--num_iterations", type=int, default=10)
    parser.add_argument("--show_progress_bar", type=bool, default=False)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--stub_agent", type=bool, default=False)
    parser.add_argument("--determine_multipliers", type=bool, default=False)
    parser.add_argument("--num_of_runs_multipliers", type=int, default=1)
    parser.add_argument("--exp_search_guidance", type=bool, default=False)
    parser.add_argument("--binary_search_guidance", type=bool, default=False)
    parser.add_argument("--decision_tree_guidance", type=bool, default=False)
    parser.add_argument("--only_exp_search", type=bool, default=False)
    parser.add_argument("--param_names", type=str, default=None)
    parser.add_argument("--runs_for_probability_estimation", type=int, default=1)
    parser.add_argument("--sb_version", type=str, default='sb2')
    parser.add_argument("--save_model", type=bool, default=False)
    parser.add_argument("--save_replay_buffer", type=bool, default=False)
    parser.add_argument("--halve_or_double", type=check_halve_or_double, default="double")
    parser.add_argument("--buffer_file", type=check_file_existence, default=None)
    parser.add_argument("--archive_file", type=check_file_existence, default=None)
    parser.add_argument("--model_suffix", type=str, default=None)

    args = parser.parse_args()

    logger = Log('search')

    param_names = None
    if args.param_names:
        try:
            param_names = args.param_names.split(sep=",")
            if len(param_names) != 2:
                raise SyntaxError('2 param names must be specified: {}'.format(args.param_names))
        except Exception:
            raise SyntaxError('param names must be comma separated: {}'.format(args.param_names))

    env_variables = instantiate_env_variables(
        algo_name=args.algo_name,
        discrete_action_space=args.discrete_action_space,
        env_name=args.env_name,
        param_names=param_names,
        model_suffix=args.model_suffix
    )
    env_eval_callback = instantiate_eval_callback(env_name=args.env_name)

    if not args.stub_agent:
        agent = Agent(
            algo_name=args.algo_name,
            env_name=args.env_name,
            log_to_tensorboard=args.log_to_tensorboard,
            tb_log_name=args.tb_log_name,
            train_total_timesteps=args.train_total_timesteps,
            n_eval_episodes=args.n_eval_episodes,
            render=args.render,
            num_envs=args.num_envs,
            model_to_load=args.model_to_load,
            continue_learning=args.continue_learning,
            discrete_action_space=args.discrete_action_space,
            eval_callback=args.eval_callback,
            env_variables=env_variables,
            continue_learning_suffix=args.continue_learning_suffix,
            env_eval_callback=env_eval_callback,
            show_progress_bar=args.show_progress_bar,
            log_every=args.log_every,
            sb_version=args.sb_version,
            save_model=args.save_model,
            save_replay_buffer=args.save_replay_buffer,
            model_suffix=args.model_suffix
        )
    else:
        agent = AgentStub(
            algo_name=args.algo_name,
            env_name=args.env_name,
            tb_log_name=args.tb_log_name,
            model_to_load=args.model_to_load,
            continue_learning=args.continue_learning,
            env_variables=env_variables,
            continue_learning_suffix=args.continue_learning_suffix,
        )

    if args.only_exp_search and args.binary_search_guidance:
        raise ValueError("only_exp_search and binary_search_guidance cannot be both True")

    alphatest = AlphaTest(
        num_iterations=args.num_iterations,
        env_variables=env_variables,
        agent=agent,
        algo_name=args.algo_name,
        env_name=args.env_name,
        tb_log_name=args.tb_log_name,
        continue_learning_suffix=args.continue_learning_suffix,
        exp_search_guidance=args.exp_search_guidance,
        binary_search_guidance=args.binary_search_guidance,
        decision_tree_guidance=args.decision_tree_guidance,
        only_exp_search=args.only_exp_search,
        buffer_file=args.buffer_file,
        archive_file=args.archive_file,
        param_names=param_names,
        runs_for_probability_estimation=args.runs_for_probability_estimation,
        determine_multipliers=args.determine_multipliers
    )
    # TODO automate this if by trying it with different environments you realize it that it works
    if args.determine_multipliers:
        multipliers, percentage_drops = alphatest.determine_multipliers(num_of_runs=args.num_of_runs_multipliers,
                                                                        halve_or_double=args.halve_or_double)
        logger.debug("Multipliers: {}".format(multipliers))
        logger.debug("Percentage drops: {}".format(percentage_drops))
    else:
        alphatest.search()
