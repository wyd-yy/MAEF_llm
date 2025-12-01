import argparse
import os
from experiment import run_llm_job, get_job_optimal_value,open_ai_keys,run_optimal_job,run_traditional_job
from llm_utils import init_logger, get_data_path
import openai
openai.api_key = ''
openai.api_base = ''

file_names = ["rue", "clu"]
algorithms = ["basic", "ec"]
batch_number = 1
llm_iteration_number = 20
pop_num = 3
plans_num = 3

def list_of_ints(arg):
    return list(map(int, arg.split(',')))

def list_of_strings(arg):
    return arg.split(',')

argParser = argparse.ArgumentParser()
argParser.add_argument("-n", "--name", help="[rue]",default=["rue"], type=list_of_strings)
argParser.add_argument("-d", "--device", default="cuda:0", type=str)
argParser.add_argument("-nc", "--node_count", default=[15], type=list_of_ints)
argParser.add_argument("-pc", "--problem_count", default=5, type=int)
argParser.add_argument("-pi", "--problem_index", default=1, type=int)
argParser.add_argument("-ki", "--key_index", default=0, type=int)
argParser.add_argument("-al", "--algorithm", default="ec_no", help="basic | ec | ec_no", type=str)
argParser.add_argument("-ad", "--adaptive", default=True, type=bool)
argParser.add_argument("-it", "--iteration", default=0, type=bool)

args = argParser.parse_args()
pi = args.problem_index
al = args.algorithm

optimal_path = get_data_path("job/job_optimal_solution.csv")
optimals = []
if not os.path.exists(optimal_path):
    run_optimal_job(["rue"], node_nums=args.node_count, problem_num=args.problem_count)
    optimals = get_job_optimal_value(op_file="job/job_optimal_solution.csv")
    run_traditional_job(["rue"], optimals=optimals, node_nums=args.node_count,
                        problem_num=args.problem_count, exe_num=1)
else:
    optimals = get_job_optimal_value(op_file="job/job_optimal_solution.csv")

for file_name in args.name:
    logger = init_logger(f"{file_name}{pi}_{al}")
    logger.info(f"experiment: {file_name}-{al}-{pi}, api key index: {args.key_index}, adaptive: {args.adaptive}")
    run_llm_job(file_name=file_name, optimals=optimals, node_nums=args.node_count, iter_num=llm_iteration_number, stop_steps=-1,
                problem_index=pi, problem_num = args.problem_count,batch_count=batch_number, api_key=open_ai_keys[args.key_index], logger=logger,
                record_file=f"{file_name}_{al}_llm_solution_fsp", algorithm=al, adaptive=args.adaptive,pop_num=pop_num,prompt_type=0,plans_num=plans_num)