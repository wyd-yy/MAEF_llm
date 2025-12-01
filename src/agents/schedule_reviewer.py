import os
import sys
from agentscope.agents import DictDialogAgent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from yaml_object_parser_new import MarkdownYAMLDictParser
from functools import partial
from agentscope.message import Msg

class ScheduleReviewer:

    HostMsg = partial(Msg, name="Moderator", role="assistant")
    def __init__(self):
        self.agent = DictDialogAgent(
            name="Advanced Scheduling Optimization Specialist",
            sys_prompt=(

                "You are a highly skilled expert in refining and optimizing job scheduling plans using genetic algorithms.Your role is to verify, repair, and improve scheduling sequences,determine the optimal makespan from the best solutions, and provide a strategic optimization guidance,and then output the best-performing schedules.\n"
                "Tasks:\n"
                "1. Evaluate and rank all feasible schedules based on defined performance metrics.\n"
                "2. Suggest improvements for genetic operations (crossover, mutation) based on population diversity and convergence rate.\n"
                "3. Provide iterative refinement guidance.\n"
                "4. Output the best-performing schedules.\n"
                "Output: \n"
                "1. Set of verified and optimized schedules with their performance metrics.\n"
                "2. Best-performing sequences.\n"
                "3. Actionable recommendations for genetic algorithm parameter adjustments.\n"
            ),
            model_config_name="kuafu3.5",
            use_memory=True
        )

        self.parser = MarkdownYAMLDictParser(
            content_hint=(
                "The Review Output should be defined in YAML format as follows:\n"
                "```yaml\n"
                "Check Result:\n"
                "Top performing valid schedules:\n"
                "Optimal makespan Value:\n"
                "Optimization Suggestion:\n"
            )
        )

        self.agent.set_parser(self.parser)

    def label_task(self, problem_detail,initial_schedules,optimized_schedules,optimal,population):
        prompt = (
            "# Problem Definition:\n"
            "```\n{problem_detail}\n```\n\n"
            "# Initial Population:\n"
            "```\n{initial_schedules}\n```\n\n"
            "# Genetic Algorithm optimized Population:\n"
            "```\n{optimized_schedules}\n```\n\n"
            "Based on this information, your task is to:\n"
            "1. Check whether the total completion time in the set of scheduling sequences includes the optimal solution time of {optimal}. If it does, directly output: 'YES' in 'Check Result'.\n"
            "2. Provide the best solution, and output the optimal makespan value.\n"
            "3. Provide one brief, strategic piece of guidance to improve the crossover and mutation operations in the next round of the genetic algorithm.\n"
            "4. From the remaining valid sequences, Directly output the top {population} performing schedules' detail.\n"
        ).format(problem_detail=problem_detail,initial_schedules=initial_schedules,optimized_schedules= optimized_schedules,optimal = optimal,population=population)
        hint = self.HostMsg(content=prompt)
        return self.agent(hint)

    def __call__(self, *args, **kwargs):
        return self.label_task(*args, **kwargs)