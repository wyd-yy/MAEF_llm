import os
import sys
from agentscope.agents import DictDialogAgent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from yaml_object_parser_new import MarkdownYAMLDictParser
from functools import partial
from agentscope.message import Msg

#ReActAgent
class ProblemDesigner:
    HostMsg = partial(Msg, name="Moderator", role="assistant")
    def __init__(self):

        self.agent = DictDialogAgent(
            name="ProblemDefiner",
            sys_prompt=(
                "You are a Problem Definer for combinatorial optimization problems. Your tasks are:\n"
                "1. Define the problem: Specify objective,decision variables, constraints, objective functions, problem type, and all jobs detail information.\n"
                "2. Design chromosome encoding: Create a simple, numerical format optimized for genetic algorithm operations.\n"
                "3. Provide an initial schedule: Generate a schedule that meets all constraints and minimizes the completion time.\n"
            ),
            model_config_name="kuafu3.5",
            use_memory=True

        )

        self.parser = MarkdownYAMLDictParser(
            content_hint={
                "The Problem defination should be responded with a YAML dictionary enclosed in XML tags as follows:\n"
                "<yaml>\n"
                "Objective:\n"
                "Decision Variables:\n"
                "Constraints:\n"
                "Objective Function:\n"
                "All Jobs Information:\n"
                "Initial Schedule:\n"
                "\n</yaml>\n"
            }
        )
        self.agent.set_parser(self.parser)

    def label_task(self, problem_detail):

        prompt = (
            "# Problem Description:\n"
            "```\n{problem_detail}\n```\n\n"
            "Based on the above information, please define the scheduling problem,and provide an initial schedule that satisfies all constraints.\n"
            """The Initial Schedule format:\n
            <trace>job_id_m,job_id_n,job_id_p,...</trace>"""
        ).format(problem_detail=problem_detail)
        hint = self.HostMsg(content=prompt)
        return self.agent(hint)

    def __call__(self, *args, **kwargs):
        return self.label_task(*args, **kwargs)