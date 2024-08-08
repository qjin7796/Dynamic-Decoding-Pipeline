import numpy as np, cv2, os, pickle
from dataclasses import dataclass
from typing import Any, Callable
from utils.init import *


def read_video(video: str) -> np.ndarray:
    if video.endswith(".mp4"):


# Map analysis names to functions
preprocessing_name_to_function: dict[str, PreprocessingMethod] = {
    "example_preprocessing": example_preprocessing,
    # ...
}


# Aliases for the type of the analysis methods: (np.ndarray, params) -> np.ndarray
# and the type of the parameters
AnalysisParameters = dict[str, Any]
AnalysisMethod = Callable[[np.ndarray, AnalysisParameters], np.ndarray]


# Define your preprocessing functions here...
def example_analysis(data: np.ndarray, params: AnalysisParameters) -> np.ndarray:
    ...


# Map analysis names to functions
analysis_name_to_function: dict[str, AnalysisMethod] = {
    "example_analysis": example_analysis,
    # ...
}


@dataclass
class DNDP:
    """Pipeline class"""
    analyses: list[str]
    params = default_params[analyses]

    def update(self, analysis_name: str, param_name: str, param_value: int) -> None:
        """Update a parameter of an analysis method."""
        # Something like that
        if analysis_name in self.analyses:
            self.params[analysis_name][param_name] = param_value
        else:
            self.params[analysis_name] = { param_name: param_value }

    def run(self) -> np.ndarray:
        """Run the pipeline and return the result."""
        self.data = self.read_input()  # Parse the inputs into the initial np.ndarray data
        self.data = self.preprocess(self.data)  # Preprocess
        self.data = self.analyse(self.data)
        return self.data

    def read_input(self) -> np.ndarray:
        data = {}
        if "video_feature" in self.analyses:
            data["video_feature"] = {}
            for video in self.params["video_feature"]["video_list"]:
                data["video_feature"][video] = read_video(video)

    def preprocess(self, data: np.ndarray) -> np.ndarray:
        # Apply the preprocessing methods sequentially to the data
        for preproc in self.preprocessing_methods:
            params = self.preprocessing_and_analysis_params[preproc.__name__]
            self.data = preproc(self.data, params)

    def analyse(self, data: np.ndarray) -> np.ndarray:
        # Apply the analysis methods sequentially to the data
        for analysis in self.analysis_methods:
            params = self.analysis_methods[analysis.__name__]
            self.data = analysis(self.data, params)


if __name__ == "__main__":
    dndp = DNDP()

    # User configures pipeline and provides input
    dndp.workflow()

    result = dndp.run()
    print(f"Pipeline result:\n{result}")
