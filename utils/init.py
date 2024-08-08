from dataclasses import dataclass
from numpy import ndarray

from typing import Any, Callable


# Aliases for the type of the preprocessing methods: (ndarray, params) -> ndarray
# and the type of the parameters
PreprocessingParameters = dict[str, Any]
PreprocessingMethod = Callable[[ndarray, PreprocessingParameters], ndarray]


# Define your preprocessing functions here...
def example_preprocessing(data: ndarray, params: PreprocessingParameters) -> ndarray:
    ...


# Map analysis names to functions
preprocessing_name_to_function: dict[str, PreprocessingMethod] = {
    "example_preprocessing": example_preprocessing,
    # ...
}


# Aliases for the type of the analysis methods: (ndarray, params) -> ndarray
# and the type of the parameters
AnalysisParameters = dict[str, Any]
AnalysisMethod = Callable[[ndarray, AnalysisParameters], ndarray]


# Define your preprocessing functions here...
def example_analysis(data: ndarray, params: AnalysisParameters) -> ndarray:
    ...


# Map analysis names to functions
analysis_name_to_function: dict[str, AnalysisMethod] = {
    "example_analysis": example_analysis,
    # ...
}


@dataclass
class DNDP:
    """Pipeline class"""

    # def __init__(self,
    #              inputs: list[ndarray],
    #              preprocessing_names: list[str],
    #              analysis_names: list[str]) -> None:

    #     self.inputs = inputs

    #     # Initialize internal data structures
    #     self.preprocessing_and_analysis_params: dict[str, dict[str, ...]] = {}
    #     self.preprocessing_methods = [ preprocessing_names[name] for name in preprocessing_names ]
    #     self.analysis_methods = [ analysis_names[name] for name in analysis_names ]

    #     # This will hold the internal intermediary data
    #     self.data: ndarray = []

    inputs: list[ndarray]
    preprocessing_and_analysis_params: dict[str, dict[str, ...]]
    preprocessing_methods: list[PreprocessingMethod]
    analysis_methods: list[AnalysisMethod]
    data: ndarray

    def update(self, preproc_or_analysis_name: str, param_name: str, param_value: int) -> None:
        """Update a parameter of a preprocessing or analysis method."""
        # Something like that
        if preproc_or_analysis_name in self.preprocessing_and_analysis_params:
            self.preprocessing_and_analysis_params[preproc_or_analysis_name][param_name] = param_value
        else:
            self.preprocessing_and_analysis_params[preproc_or_analysis_name] = { param_name: param_value }

    def run(self) -> ndarray:
        """Run the pipeline and return the result."""
        self.data = self.read_input()  # Parse the inputs into the initial ndarray data
        self.data = self.preprocess(self.data)  # Preprocess
        self.data = self.analyse(self.data)
        return self.data

    def read_input(self) -> ndarray:
        # TODO: parse self.inputs into the initial ndarray data
        ...

    def preprocess(self, data: ndarray) -> ndarray:
        # Apply the preprocessing methods sequentially to the data
        for preproc in self.preprocessing_methods:
            params = self.preprocessing_and_analysis_params[preproc.__name__]
            self.data = preproc(self.data, params)

    def analyse(self, data: ndarray) -> ndarray:
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
