import os
import csv
from llm_utils import write_csv_file, get_data_path, init_logger, get_optimal_value,get_job_optimal_value, open_ai_keys
from llm_tsp import evaluate_job

# python src/executor.py -n rue,clu -nc 10,15 -pi 1 -al ec
job_aglorithms = ["spt", "ga","hybrid_Ga"]
job_llm = ["kuafu_max_v3.6"]




def run_llm_job(file_name, optimals, node_nums, iter_num, stop_steps, problem_index, problem_num,batch_count, api_key, logger,
                record_file="job_llm_solution", system=None, random_select=True, device="cuda:0", algorithm="basic",
                adaptive=False,
                start_iter=0, init_population=None, pop_num=4, prompt_type=0,plans_num=4):
    out_path = get_data_path("output")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    header = ["batch", "model", "node number", "problem", "distance", "optimal distance", "gap", "success_step",
              "route"]
    sol_path = write_csv_file(file_path=f"output/{record_file}_{problem_index}.csv", header=header, overwrite=False)
    iter_header = ["batch", "model", "node number", "problem", "iteration", "distance", "optimal distance", "gap"]
    iter_path = write_csv_file(file_path=f"output/{record_file}_iter_{problem_index}.csv", header=iter_header,
                               overwrite=False)
    for bt in range(batch_count):
        for model in job_llm:
            for j in node_nums:
                for i in range(problem_num):
                    success_step = "NA"
                    steps, gap, min_dist, route, _, _ = evaluate_job( iter_log_path=iter_path, model=model,
                                                                 api_key=api_key, iter_num=iter_num,
                                                                 logger=logger,
                                                                 stop_steps=stop_steps, system_msg=system, pop_num=pop_num,
                                                                 node_num=j, problem_index=problem_index,problem_num=i+1,
                                                                 batch_index=bt + 1,
                                                                 random_select=random_select, device=device,
                                                                 algorithm=algorithm, adaptive=adaptive,
                                                                 start_iter=start_iter,
                                                                 init_population=init_population, prompt_type=prompt_type,isgpt = False,plans_num=plans_num)
                    if steps != -1:
                        success_step = str(steps)
                    with open(sol_path, "a", encoding="UTF8", newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(
                            [str(bt + 1), model, str(j), str(i+1), str(min_dist), "{:.3f}".format(gap),
                             success_step, route])
                    print(bt,model,j,i)



def run_optimal_job(file_names, node_nums, problem_num):
    header = ["file", "node number", "problem", "distance", "route"]
    data = []
    result = [[20,18,17,16,13],[32,25,23,22,19],[47,36,33,30,28]]
    for file_name in file_names:
        for j in node_nums:
            for i in range(problem_num):
                sol = result[node_nums.index(j)][i]
                data.append([file_name, str(j), str(i + 1), str(sol),
                             ""])
    write_csv_file("job/job_optimal_solution.csv", header, data)

def run_traditional_job(file_names, optimals, node_nums, problem_num, exe_num=1):
    header = ["batch", "file", "algorithm", "node number", "problem", "distance", "optimal distance", "gap"]
    result = [[[20,18,17,16,13],[35,32,26,24,22],[63,51,39,46,32]],[[42,40,37,35,37],[46,44,44,41,37],[66,53,49,45,43]],[[20,18,17,16,13],[33,27,25,22,21],[52,40,38,34,29]]]
    data = []
    for file_name in file_names:
        for alg in job_aglorithms:
            for j in node_nums:
                for i in range(problem_num):
                    for et in range(exe_num):
                        distance = result[job_aglorithms.index(alg)][node_nums.index(j)][i]
                        op = optimals[f"{file_name}-{j}-{i + 1}"]
                        gap = "{:.3f}".format((distance - op) / op * 100)
                        data.append([str(et + 1), file_name, alg, str(j), str(i + 1), str(int(distance)), str(op), gap])
    write_csv_file("job/job_tradition_solution.csv", header, data)
