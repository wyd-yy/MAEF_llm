import os
import sys
from agentscope.agents import DictDialogAgent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from yaml_object_parser import MarkdownYAMLDictParser
from functools import partial
from agentscope.message import Msg

class GenerateSchedule:
    HostMsg = partial(Msg, name="Moderator", role="assistant")
    def __init__(self):
        self.agent = DictDialogAgent(
            name="Scheduling Optimization Expert",
            sys_prompt=(
                "You are a high-level scheduling optimization expert tasked with generating and validating an initial scheduling plan based on the following problem description. Your goal is to produce a schedule that satisfies all constraints and minimizes total completion time.\n"
            ),
            model_config_name="kuafu3.5",
            use_memory=True
        )

        self.parser = MarkdownYAMLDictParser(
            content_hint=(
                "The Generate schedule should be defined in YAML format as follows:\n"
                "```yaml\n"
                "Final Scheduling Plan:\n"
                "Validation Results:\n"
            )
        )
        self.agent.set_parser(self.parser)

    def label_task(self, problem_detail):
        prompt = (
            "# Problem Definition:\n"
            "```\n{problem_detail}\n```\n\n"
            "###Task:\n"
            "1.Generate an Initial Scheduling Plan:\n"
            " - Create an initial scheduling plan that includes the start and end times for each task.\n"
            "2.Validate the Scheduling Plan:\n"
            " - After generating the scheduling plan, write Python code to verify that the plan meets all constraints. If the validation fails, regenerate the scheduling plan and revalidate until a valid schedule that meets all constraints is produced.\n"
            "3.Calculate and Output Total Completion Time:\n"
            " - Calculate the total completion time (makespan) for the scheduling plan. Output the final scheduling plan along with the validation results.\n"

        ).format(problem_detail=problem_detail)
        hint = self.HostMsg(content=prompt)
        return self.agent(hint)

    def __call__(self, *args, **kwargs):
        return self.label_task(*args, **kwargs)