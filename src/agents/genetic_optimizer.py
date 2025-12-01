import os
import sys
from agentscope.agents import DictDialogAgent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from yaml_object_parser_new import MarkdownYAMLDictParser
from functools import partial
from agentscope.message import Msg

#ReActAgent
class GeneticOptimizer:

    HostMsg = partial(Msg, name="Moderator", role="assistant")
    def __init__(self):
        self.agent = DictDialogAgent(
            name="Genetic Algorithm Evolution Specialist",
            sys_prompt=(
                "You are an expert in genetic algorithms for job scheduling, specializing in executing and optimizing evolutionary operations.Your task is to generate new scheduling plans based on the given problem definition and initial population. Follow these steps:\n"
                "1. Input: Receive problem definition (jobs, resources, constraints) ,initial population of schedules and optimization suggestions.\n"
                "2. Iterative process:\n"
                "a) Select parent schedules.\n"
                "b) Apply crossover and mutation to create offspring.\n"
                "3. Output: Return improved scheduling plans after specified iterations or convergence.\n"
                "Execute this process and provide the resulting schedules.\n"
            ),
            model_config_name="kuafu3.5",
            use_memory=True

        )

        self.parser = MarkdownYAMLDictParser(
            content_hint={
                "The Optimized Schedules should be defined in YAML format as follows:\n"
                "```yaml\n"
                "Optimized Schedules:\n"

            },
            fix_model_config_name = "kuafu3.5"
        )
        self.agent.set_parser(self.parser)

    def label_task(self, problem_detail,population,initial_schedules,optimization_suggestions):

        prompt = (
            "# Problem Definition:\n"
            "```\n{problem_detail}\n```\n\n"
            "# Initial Population:\n"
            "```\n{initial_schedules}\n```\n\n"
            "# Optimization Suggestions:\n"
            "```\n{optimization_suggestions}\n```\n\n"
            """The schedules format:\n
            <trace>job_id_m,job_id_n,job_id_p,...</trace>\n
            <trace>job_id_m,job_id_n,job_id_p,...</trace>\n
            """
        ).format(problem_detail=problem_detail,population = population,initial_schedules=initial_schedules,optimization_suggestions= optimization_suggestions)
        hint = self.HostMsg(content=prompt)
        return self.agent(hint)

    def __call__(self, *args, **kwargs):
        return self.label_task(*args, **kwargs)