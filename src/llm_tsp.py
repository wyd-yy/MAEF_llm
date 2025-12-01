import random
import time

from langchain.chains import LLMChain

#OPRO
def llm_process_job_kuafu(llm, tokens, secs, points, population, logger,total_resource,plans_num):
    content = " ".join([x[0] for x in population])
    logger.info(f"content: {content}")
    prompt_template = "you are an expert in job scheduling problems."

    chain = LLMChain(llm=llm, prompt=prompt_template)
    messages = [
        {"role": "user", "content": f"You are given a list detailing the information for various scheduling jobs, each with distinct processing times,resource demands and predecessors. All jobs necessitate the same type of resource, with a total available resource pool of {total_resource}.{points}."
            f"Below are some previous scheduling plans and their total completion times,each scheduling plan start with <trace> and end with </trace>, and each job records their id,start time and finish time. The scheduling plans are arranged in descending order based on their total completion times, where lower values are better.{content}"
            f"Give me at least {plans_num} new scheduling plans that are different from all scheduling plans above, and each scheduling plan has a total completion time lower than any of the above. Each scheduling plan should complete all jobss exactly once,and must satisfy the Precedence Constraints."
            "The scheduling plan should start with <trace> and end with </trace>,no explanation."}
    ]
    try:
        result = chain.run(messages=messages)
        response = result.get('response', {})
        generated_message = response.get('content', [{'text': 'content_error'}])[0].get('text', 'text_error')
    except Exception as e:
        print(f"error: {str(e)}")
    logger.info(f"res: {generated_message}")
    return generated_message, tokens

# genetic prompt
def get_cm_prompts(prompt_type):
    crossover_msg1 = "There are 2 different crossover operators you can use: PMX (Partially Mapped Crossover), OX (Ordered Crossover)"
    mutation_msg1 = "There are 3 different mutation operators you can use: Swap Mutation, Insert Mutation, Inversion Mutation"
    crossover_msg2 = '''There are 2 different crossover operators you can use:
        1. PMX (Partially Mapped Crossover):
            - Description: PMX randomly selects a segment from parent 1, copies it to the offspring, and fills in the remaining positions of the offspring by mapping elements from parent 2.
            Below is an example.
                - Parent 1: 1 2 3 4 5 6 7 8
                - Parent 2: 3 7 5 1 6 8 2 4
                - Randomly select a segment from parent 1 (e.g., positions 4 to 6): 4 5 6
                - Copy the segment from Parent 1 to offspring solution: _ _ _ 4 5 6 _ _ 
                - Fill in the remaining positions by mapping elements from parent 2 (note elements cannot be repeated) to the offspring: 3 7 8 4 5 6 2 1
        2. OX (Ordered Crossover):
            - Description: OX randomly selects a segment from parent 1, copies it to the offspring, and fills in the remaining positions with the missing elements in the order in which they appear in parent 2.
            Below is an example.
                - Parent 1: 1 2 3 4 5 6 7 8
                - Parent 2: 3 7 5 1 6 8 2 4
                - Randomly select a segment from parent 1 (e.g., positions 4 to 6): 4 5 6
                - Copy the segment from Parent 1 to the offspring: _ _ _ 4 5 6 _ _ 
                - The missing elements in the order in which they appear in parent 2 are {3, 7, 1, 8, 2}
                - Fill in the remaining positions of the offspring based on the above sorted elements: 3 7 1 4 5 6 8 2'''
    mutation_msg2 = '''There are 3 different mutation operators you can use:
        1. Swap Mutation:
            - Description: swap mutation randomly selects two positions in an individual and swaps the elements at those two positions.
            - Example:
                - original: 5 2 8 4 1 7 6 3
                - Randomly select two positions, e.g., position 3 and posision 6
                - Swap the elements 8 and 7 at position 3 and position 6: 5 2 7 4 1 8 6 3
        2. Insert Mutation:
            - Description: insert mutation randomly selects one position in the individual and moves the element at that position to another randomly chosen position.
            - Example:
                - original: 5 2 8 4 1 7 6 3
                - Randomly select one position, e.g., position 3
                - Move the element 8 at position 3 to another randomly chosen position 6: 5 2 4 1 7 8 6 3
        3. Inversion Mutation:
            - Description: inversion mutation randomly selects two positions in an individual and inverts the order of the elements between those positions.
            - Example:
                - original: 5 2 8 4 1 7 6 3
                - Randomly select two positions, e.g., position 3 and posision 6
                - inverts the order of the elements between position 3 and position 6: 5 2 7 1 4 8 6 3'''
    crossover_selection = "Select one of the crossover operators based on above EC knowledge , use the selected crossover operator to crossover"
    mutation_selection = "Select one of the Mutation operators based on above EC knowledge, use the selected crossover operator to mutate"

    if prompt_type == 0:
        return crossover_msg1, crossover_selection, mutation_msg1, mutation_selection
    elif prompt_type == 1:
        return crossover_msg2, crossover_selection, mutation_msg2, mutation_selection
    elif prompt_type == 2:
        return crossover_msg2, crossover_selection, "", "Mutate"
    elif prompt_type == 3:
        return "", "Crossover", mutation_msg2, mutation_selection


def get_cm_job_prompts(prompt_type):
    crossover_msg1 = "There are 2 different crossover operators you can use: PMX (Partially Mapped Crossover), TWX (Time Window Crossover)"
    mutation_msg1 = "There are 3 different mutation operators you can use: Task Swap Mutation, Task Sliding Mutation, Time Perturbation Mutation"
    crossover_msg2 = '''There are 2 different crossover operators you can use:
        1. PMX (Partially Mapped Crossover):
            Steps:
            1.Select Crossover Points: Randomly select two crossover points from the parent chromosomes.
            2.Swap Segments: Swap the segments between the two crossover points from both parents, concerning task IDs.
            3.Resolve ID Conflicts: Check for any duplicate task IDs in the offspring chromosomes. For each duplicate, find a non-duplicate ID from the original parents and replace it.
            4.Copy Times: Copy the start and end times from the corresponding tasks in the parents to the appropriate tasks in the offspring.
            Conflict Resolution:
            Ensure that the replacement process does not introduce new duplicate IDs.
            Check time constraints to ensure that tasks still comply with the resource constraints and dependencies of the jobs.
        2. TWX (Time Window Crossover):
            Steps:
            1.Define Time Window: Choose a time range as the crossover window.
            2.Swap Times: Within the selected time window, swap the start and end times of the tasks between the two parent chromosomes.
            3.Time Verification: Verify that the swapped times meet the dependencies and resource constraints of the tasks.
            Conflict Resolution:
            If swapped times do not meet constraints, adjust the times slightly or revert to the original parents’ time configuration.'''
    mutation_msg2 = '''There are 3 different mutation operators you can use:
        1. Task Swap Mutation:
            Steps:
            1.Select Task Pair: Randomly select two tasks.
            2.Swap Tasks: Exchange their IDs, start times, and end times.
            3.Verify Schedule: Check if the new time arrangement complies with task dependencies and resource constraints.
        2. Task Sliding Mutation:
            Steps:
            1.Select Task: Randomly pick a task.
            2.Slide Task: Slide the task forward or backward on the timeline.
            3.Legality Verification: Ensure that the new timing does not violate any constraints.
        3. Time Perturbation Mutation:
            Steps:
            1.Select Task: Randomly select a task.
            2.Adjust Times: Make slight adjustments to the start and end times of the task.
            3.Check Legality: Verify that the adjusted times comply with all related constraints.'''
    crossover_selection = "Select one of the crossover operators based on above EC knowledge , use the selected crossover operator to crossover"
    mutation_selection = "Select one of the Mutation operators based on above EC knowledge, use the selected mutation operator to mutate"

    if prompt_type == 0:
        return crossover_msg1, crossover_selection, mutation_msg1, mutation_selection
    elif prompt_type == 1:
        return crossover_msg2, crossover_selection, mutation_msg2, mutation_selection
    elif prompt_type == 2:
        return crossover_msg2, crossover_selection, "", "Mutate"
    elif prompt_type == 3:
        return "", "Crossover", mutation_msg2, mutation_selection

#lmea
def llm_process_job_ec_kuafu(llm, tokens, secs, points, population, hints, logger, prompt_type=0, random_select=False,total_resource=20,plans_num=8):
    all_population = " ".join([x[0] for x in population])
    self_hints = ""
    if len(hints) > 0:
        self_hints = "Here are the hints you should follow to choose corssover operator and mutation operator:" + "\n".join(
            [x for x in hints])
    logger.info(f"content: {all_population}, hint: {self_hints}")

    crossover, crossover_selection, mutation, mutation_selection = get_cm_job_prompts(prompt_type=prompt_type)

    prompt_template = "You are an evolutionary computing expert for the Jobshop Scheduling Problem."

    chain = LLMChain(llm=llm, prompt=prompt_template)

    messages = [
        {"role": "user", "content": f'''You are given a list of scheduling jobs' information, each with distinct processing times, resource demands, and predecessors. All jobs require the same type of resource, with a total available resource pool of {total_resource}.
        Here are the jobs' information {points}.
        Also,you are given some previous scheduling plans and their total completion times,each scheduling plan start with <trace> and end with </trace>, and each job records their id,start time and finish time. The scheduling plans are arranged in descending order based on their total completion times, where lower values are better.{all_population}"
        EC knowledge: {crossover}\n{mutation}\n

        You should follow the below instruction step-by-step to generate new scheduling plans from given jobs and scheduling plans. 
        {self_hints}
        Ensure you preserve selected corssover operator in Step 2, selected mutation operator in Step 3, and the scheduling plans at each step, repeat Step 1, 2, 3 for a given iteration number.
        1. choose any two scheduling plans from the given scheduling plans, and save the two chosen scheduling plans, bracketed them with <sel> and </sel>.
        2. {crossover_selection} the two scheduling plans got in Step 1 and generate a new scheduling plan that is different from all scheduling plans, and has a total completion time lower than any of these two scheduling plans. 
        The generated scheduling plan should contains all jobs exactly once. Save the selected crossover operator and bracketed it with <c> and </c>. Save the generated scheduling plan and bracketed it with <cross> and </cross>.
        3. {mutation_selection} the scheduling plan generated in Step 2 and generate a new scheduling plan that is different from all scheduling plans, and has a lower total completion time.
        The scheduling plan should contains all jobs exactly once. Save the selected mutation operator and bracketed it with <m> and </m>. Save the generated scheduling plan and bracketed it with <trace> and </trace>.

        Directly give me all the saved selected crossover operator from Step 2, the mutation operator from Step 3, and the scheduling plans from each Step without any explanations.
        The output format should be similiar with below, and the output should contain 16 iterations:
        Iteration 1:
        Step 1: <sel>[1, 0, 0],[2, 0, 5],[5, 12, 13],[3, 10, 15],[4, 10, 15],[7, 19, 20],[6, 21, 22],[8, 21, 23],[10, 22, 24],[9, 24, 26],[11, 23, 30],[12, 26, 33],[14, 33, 38],[13, 33, 40],[15, 43, 48],[16, 46, 51],[17, 51, 51]</sel>, <sel>[1, 0, 0],[3, 0, 5],[6, 9, 10],[4, 7, 12],[2, 10, 15],[9, 13, 15],[5, 22, 23],[7, 22, 23],[12, 18, 25],[10, 28, 30],[8, 29, 31],[15, 29, 34],[11, 31, 38],[13, 35, 42],[14, 39, 44],[16, 46, 51],[17, 51, 51]</sel>
        Step 2: <c>PMX (Partially Mapped Crossover)</c><cross>[1, 0, 0],[3, 3, 8],[6, 10, 11],[4, 6, 11],[2, 9, 14],[7, 16, 17],[10, 17, 19],[9, 19, 21],[5, 20, 21],[8, 24, 26],[12, 21, 28],[15, 28, 33],[13, 29, 36],[11, 34, 41],[16, 37, 42],[14, 45, 50],[17, 50, 50]</cross>
        Step 3: <m>Task Swap Mutation</m><trace>[1, 0, 0],[3, 0, 5],[4, 4, 9],[6, 9, 10],[2, 8, 13],[7, 17, 18],[9, 19, 21],[5, 23, 24],[10, 22, 24],[8, 26, 28],[12, 29, 36],[11, 31, 38],[13, 31, 38],[14, 39, 44],[16, 39, 44],[15, 45, 50],[17, 50, 50]</trace>
        Iteration 2:
        Step 1: <sel>[1, 0, 0],[4, 1, 6],[3, 2, 7],[2, 4, 9],[5, 12, 13],[7, 14, 15],[6, 14, 15],[8, 19, 21],[9, 20, 22],[10, 20, 22],[13, 26, 33],[12, 27, 34],[11, 28, 35],[16, 35, 40],[14, 38, 43],[15, 44, 49],[17, 49, 49]</sel>, <sel>[1, 0, 0],[2, 1, 6],[3, 5, 10],[5, 10, 11],[6, 13, 14],[4, 9, 14],[7, 17, 18],[8, 17, 19],[10, 18, 20],[9, 20, 22],[11, 19, 26],[13, 23, 30],[14, 30, 35],[12, 30, 37],[16, 38, 43],[15, 44, 49],[17, 49, 49]</sel>
        Step 2: <c>TWX (Time Window Crossover)</c><cross>[1, 0, 0],[4, 0, 5],[2, 3, 8],[7, 7, 8],[5, 12, 13],[3, 9, 14],[10, 12, 14],[8, 13, 15],[6, 21, 22],[13, 15, 22],[9, 24, 26],[11, 22, 29],[16, 24, 29],[12, 33, 40],[14, 37, 42],[15, 43, 48],[17, 48, 48]</cross>
        Step 3: <m>Task Sliding Mutation</m><trace>[1, 0, 0],[3, 1, 6],[4, 3, 8],[7, 8, 9],[2, 8, 13],[5, 15, 16],[6, 15, 16],[9, 18, 20],[10, 19, 21],[8, 26, 28],[13, 25, 32],[12, 28, 35],[11, 31, 38],[15, 37, 42],[16, 40, 45],[14, 43, 48],[17, 48, 48]</trace>
        '''}
    ]
    try:
        result = chain.run(messages=messages)
        response = result.get('response', {})

        generated_message = response.get('content', [{'text': 'content_error'}])[0].get('text', 'text_error')
    except Exception as e:
        print(f"error: {str(e)}")
    logger.info(f"res: {generated_message}")
    return generated_message, tokens

from utils import *
#maef
def llm_process_job_ec_no_kuafu(tokens,logger,plans_num=8,iters_num=1,opt_sug = None,sug_str=None,result_str = None):
    """A basic conversation demo"""

    configs = agentscope.init(
        model_configs="./configs/model_configs_template.json"
    )
    config = configs[3]
    ag = AgentGroups("./agents")

    jobs = {
        1:[3, 2, 2],
        2:[2, 1, 4],
        3:[4, 3, 3]
    }
    #description
    problem_detail = """Here is a Flow Shop Scheduling Problem,There are 3 machines M1,M2,M3 and 10 jobs J1-J10 with processing times J1=(3, 2, 2),J2=(2, 1, 4),J3=(4, 3, 3),minimize the makespan (the total time to complete all jobs)"""
    #optimal_value
    optimal = 1



    plans_num = 8
    pop_num = plans_num


    result_str = ag.define_problem(problem_detail, pop_num, iters_num, opt_sug, sug_str,
                                   result_str, config, optimal)
    result = result_str

    return result,tokens





def evaluate_job(iter_log_path, model, api_key, node_num, problem_index, problem_num, logger, iter_num=10,
             pop_num=4, stop_steps=-1, system_msg=None,
             batch_index=1, random_select=True, algorithm="basic", device="cuda:0", adaptive=False, start_iter=0,
             init_population=None, prompt_type=0,isgpt = False,plans_num = 4):
    success_step = -1
    min_dist = float("inf")
    start_time = datetime.now()
    short_routes, route, worse_iter, tokens, gap, iter_result = [], "", 0, 0, float('inf'), []
    sug_str = ""
    opt_sug = ""
    result_str = ""
    logger.debug(f"<<<<==========start LLM tsp => {model}, {iter_num}, {pop_num}, {stop_steps}==========>>>>")
    for i in range(start_iter, iter_num):
        secs = (datetime.now() - start_time).total_seconds()
        if secs >= 60:
            secs = 60
            tokens = 0
            start_time = datetime.now()
        res, tokens = llm_process_job_ec_no_kuafu(tokens=tokens,logger=logger, plans_num=plans_num,iters_num = i,opt_sug = opt_sug,sug_str = sug_str,result_str = result_str)
        time.sleep(0.4)

    return success_step, gap, min_dist, route, init_population

