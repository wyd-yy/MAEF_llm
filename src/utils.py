import os
import re
from collections import defaultdict
import json
import importlib
from functools import partial
import agentscope
from datetime import datetime

from agentscope.message import Msg
import random


class AgentGroups:
    
    HostMsg = partial(Msg, name="Moderator", role="assistant")
    dumps_json = partial(json.dumps, separators=(',', ':'), indent=None, ensure_ascii=False)
    def __init__(self, agents_dir='agents'):
        self.agents = {}
        for filename in os.listdir(agents_dir):
            if filename != 'problem_designer.py' and filename != 'genetic_optimizer.py' and filename != 'schedule_reviewer.py' and filename != 'initial_population.py':
                continue
            # if filename != 'problem_designer.py':
            #     continue
            if filename.endswith('.py') and not filename.startswith('__'):
                print(f"Importing {filename}")
                module_name = filename[:-3]
                module = importlib.import_module(f'agents.{module_name}')
                class_name = ''.join(word.capitalize() for word in module_name.split('_'))
                if hasattr(module, class_name):
                    agent_class = getattr(module, class_name)
                    self.agents[class_name] = agent_class()
    
    def get_agent(self, agent_name):
        return self.agents.get(agent_name)

    def define_problem(self, problem,pop_num,iters_num,opt_sug,sug_str,result_str,config,optimal,jobs):
        iters = 8
        optimal_result = optimal
        min_dist = 9999
        same_iter = 0
        for iter in range(iters):
            if iter == 0:
                problemdesigner = self.get_agent('ProblemDesigner')
                start_time = datetime.now()
                result = problemdesigner(problem)
                secs = (datetime.now() - start_time).total_seconds()
                result_str = result.content

                initialpopulation = self.get_agent('InitialPopulation')
                start_time = datetime.now()
                inital_schedules = initialpopulation(problem_detail = result_str,
                                                 population = pop_num,
                                                     optimal = optimal_result)
                secs = (datetime.now() - start_time).total_seconds()
                print("secs：", secs)
                initial_str = inital_schedules.content.get("Initial Population")

                sols = self.get_solutions_fsp(jobs,str(inital_schedules.content))
                if len(sols) != 0:
                    traces = self.evl_process_job_fsp(jobs,sols)
                    traces.sort(key=lambda x: x[1], reverse=True)
                    min_dist = traces[-1][1]
                    check_str = min_dist
                    if min_dist == optimal_result:
                        break
                traces = sols
                initial_str = str(traces)
                if check_str < optimal_result:
                    opt_sug = f"The schedule with the minimum total completion time among the above schedules violates the No overlap machine constraints and is therefore invalid.Please consider this scheduling sequence as a failure case.\n"
                else:
                    opt_sug = "Please generate new scheduling plans with the lowest possible total completion time.\n"
            else:
                initial_str = traces_dict

            geneticoptimizer = self.get_agent('GeneticOptimizer')
            start_time = datetime.now()
            opt_schedules = geneticoptimizer(problem_detail=result_str,
                                        population=pop_num,
                                        initial_schedules = initial_str,
                                        optimization_suggestions = opt_sug)
            secs = (datetime.now() - start_time).total_seconds()
            print("GeneticOptimizer：", secs)
            opt_str = opt_schedules.content
            sols1 = self.get_solutions_fsp(jobs, str(opt_str))
            # if min_dist == optimal_result:
            #     break
            if len(sols1) != 0:
                traces1 = self.evl_process_job_fsp(jobs, sols1)
                traces1.sort(key=lambda x: x[1], reverse=True)
                min_dist = traces1[-1][1]
                if min_dist == optimal_result:
                    break
            opt_str = str(sols1)
            schedulereviewer = self.get_agent('ScheduleReviewer')
            start_time = datetime.now()
            suggestions = schedulereviewer(problem_detail = result_str,
                                           initial_schedules = initial_str,
                                           optimized_schedules=opt_str,
                                           optimal = optimal_result,
                                           population=pop_num)
            secs = (datetime.now() - start_time).total_seconds()
            repaired_plans = suggestions.content.get("Check Result")
            opt_sug = suggestions.content.get("Optimization Suggestion")

            min_makespan = min_dist
            if min_dist < optimal_result:
                opt_sug = f"The schedules with the minimum total completion time among the above schedules violate the No overlap machine constraints and are therefore invalid.Please consider these scheduling sequences as failure cases.\n"
                traces_dict = initial_str
                continue
            if min_makespan >= min_dist and min_makespan > optimal_result:
                same_iter += 1
            elif min_makespan < min_dist and min_makespan > optimal_result:
                min_dist = min_makespan
                same_iter = 0
            if same_iter >= 2:
                opt_sug = f"After observing no improvement over the last {same_iter} iterations, the genetic algorithm may be stuck in a local optimum({int(min_makespan)}).Here are some strategies to guide these operations:Diverse Crossover Operators,Targeted Mutation,Diversity Preservation,Elitism with Diversity.\n"
                same_iter = 0
                if config['generate_args']['temperature'] < 1.3:
                    config['generate_args']['temperature'] += 0.1
                    config['generate_args']['max_tokens'] = 90000
                    agentscope.init(
                        model_configs=config
                    )
                    if pop_num < 10:
                        pop_num += 1


            traces_dict = suggestions.content.get("Top performing valid schedules")

        return result_str

    def get_solutions_fsp(self,jobs, content: str):
        import re
        def check_schedule_constraints(schedule):
            if len(schedule) == len(jobs) and all(job in schedule for job in jobs):
                return True
            return False

        def parse_trace_string(trace_string):
            pattern = r"<trace>(.*?)</trace>"
            matches = re.findall(pattern, trace_string)

            valid_schedules = []

            for match in matches:
                schedule = list(map(int, match.split(',')))
                if check_schedule_constraints(schedule):
                    valid_schedules.append(schedule)

            return valid_schedules

        valid_schedules = parse_trace_string(content)

        return valid_schedules

    def evl_process_job_fsp(self,jobs, schedules):

        results = []
        num_jobs = len(jobs)
        num_machines = len(next(iter(jobs.values())))

        for schedule in schedules:
            machine_times = [0] * num_machines

            for job in schedule:
                job_times = jobs[job]
                for i in range(num_machines):
                    if i == 0:
                        machine_times[i] += job_times[i]
                    else:
                        machine_times[i] = max(machine_times[i], machine_times[i - 1]) + job_times[i]

            total_completion_time = machine_times[-1]
            print(schedule,total_completion_time)

            results.append(
                (f"<trace>{','.join(map(str, schedule))}</trace>,total_completion_time:{total_completion_time}",
                 total_completion_time))
        return results

