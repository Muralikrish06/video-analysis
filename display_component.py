from langflow.custom import Component
from langflow.io import DataInput, Output
from langflow.schema import Data


class InsurancePolicyDisplayComponent(Component):
    display_name = "Insurance Policy Display"
    description = "Returns insurance policy verification results as structured JSON."
    icon = "file-text"
    name = "InsurancePolicyDisplay"

    inputs = [
        DataInput(
            name="policy_data",
            display_name="Policy Data",
            info="The JSON output from the policy extraction component.",
        ),
    ]

    outputs = [
        Output(
            display_name="Policy JSON",
            name="policy_json",
            method="build_output",
        ),
    ]

    def build_output(self) -> Data:
        data = self.policy_data.data if isinstance(self.policy_data, Data) else {}
        return Data(data=data)