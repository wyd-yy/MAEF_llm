import os
import sys
from agentscope.agents import DictDialogAgent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from yaml_object_parser_new import MarkdownYAMLDictParser
from functools import partial
from agentscope.message import Msg

#ReActAgent
class InitialPopulation:
    HostMsg = partial(Msg, name="Moderator", role="assistant")
    def __init__(self):
        self.agent = DictDialogAgent(
            name="Expert Scheduling Optimizer",
            sys_prompt=(
                #调度问题
                "As an expert in genetic algorithms for job scheduling, generate a diverse, high-quality initial population of feasible scheduling plans.\n"
                "Tasks:\n"
                "1. Create diverse, feasible schedules meeting all constraints.\n"
                "2. Optimize for the objective function while maintaining diversity.\n"
                "3. Encode plans in a genetic algorithm-compatible format.\n" #including task IDs, start and end times.\n"
                "4. Evaluate and assign fitness to each plan step by step.\n"
                "Output: A set of unique schedules, each with its makespan value.\n"

            ),
            model_config_name="kuafu3.5",
            use_memory=True
        )

        self.parser = MarkdownYAMLDictParser(
            content_hint={
                "The Initial Population should be responded with a YAML dictionary enclosed in XML tags as follows:\n"
                "```yaml\n"
                "Initial Population:\n"
                "Check Result:\n"
            },
            fix_model_config_name = "kuafu3.5"
        )

        self.agent.set_parser(self.parser)

    def label_task(self, problem_detail,population,optimal):
        prompt = (
            "# Problem Definition:\n"
            "```\n{problem_detail}\n```\n\n"
            "Based on this information, please generate at least {population} distinct initial schedules for the scheduling problem.\n"
            """The Initial Schedule format:\n
            <trace>job_id_m,job_id_n,job_id_p,...</trace>"""
            "Please directly output the mean total completion time of these scheduling sequences under 'Check Result', no explanation.\n"

        ).format(problem_detail=problem_detail,population = population,optimal = optimal)
        hint = self.HostMsg(content=prompt)
        return self.agent(hint)

    def __call__(self, *args, **kwargs):
        return self.label_task(*args, **kwargs)