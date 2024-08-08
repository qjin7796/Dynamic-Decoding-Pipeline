from dataclasses import dataclass
from typing import Any, Callable
from utils import *

@dataclass
class DNDP:
    """Pipeline class"""
    analyses: list[str] | dict[str, Any] 
    if isinstance(analyses, list):
        params = init.default_params[analyses]
    else:
        analyses = analyses.keys()
        params = analyses
    # Map analysis names to functions
    analysis_name_to_function: dict[str, Callable] = {
        ...
    }

    def configure(self, analysis_name: str, param_name: str, param_value: int) -> None:
        """Configure parameters of analysis methods."""
        # Something like that
        if analysis_name in self.analyses:
            self.params[analysis_name][param_name] = param_value
        else:
            self.params[analysis_name] = { param_name: param_value }

    def run(self) -> np.ndarray:
        """Run the pipeline and return the result."""
        input = self.read_input()  # Parse the inputs into the initial np.ndarray data
        data = self.preprocess(input)  # Preprocess
        output = self.analyze(data)
        return output

    def read_input(self) -> np.ndarray:
        data = {}
        if "video_feature" in self.analyses:
            data["video_feature"] = {}
            for video in self.params["video_feature"]["video_list"]:
                data["video_feature"][video] = read_video(video)

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        # Apply the preprocessing methods sequentially to the data
        ...

    def analyze(self, data: np.ndarray) -> np.ndarray:
        # Apply the analysis methods sequentially to the data
        ...

