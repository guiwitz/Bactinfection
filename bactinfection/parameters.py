
from dataclasses import dataclass
import dataclasses
from pathlib import Path
import yaml

@dataclass
class Param:
    """
    Class for keeping track of processing parameters.
    
    Paramters
    ---------
    data_folder: str
        main folder containing oir files or sub-folders with oir files
    data_folder: str
        folder containing analysed files
    bact_len: int
        estimated bacteria length for template matching
    bact_width: int
        estimated bacteria width for template matching
    diameter_nucl: int
        estimated nuclei diameter
    n_std: float
        number of standard deviation beyond background intensity for
        bacteria segmentation
    corr_threshold: float
        threshold in range [0, 1] for template matching quality (1 is best)
    min_corr_vol: float
        minimal number of voxesl above corr_threshold (designed to exclude
        single very bright spots)
    nucl_model_type: str
        cellpose model to use for nuclei segmentation (nuclei or cyto)
    masking: str
        keep bacteria under mask "nuclei", "cells" or "cells_no_nuclei"
    
    """
    data_folder: str
    analysis_folder: str = None
    nucl_channel: str = None
    cell_channel: str = None
    bact_channel: str = None
    bact_len: int = 5
    bact_width: int = 5
    diameter_nucl: int = 50
    n_std: float = 10
    corr_threshold: float = 0.5
    min_corr_vol: int = 5
    nucl_model_type: str = 'nuclei'
    masking: str = 'nuclei'

    def __post_init__(self):
        self.data_folder = Path(self.data_folder).resolve()

    def save_parameters(self, data_path=None):
        """Save parameters as yml file.

        Parameters
        ----------
        data_path : str or Path, optional
            path to data folder, by default none
        """

        self.analysis_folder = Path(self.analysis_folder).resolve()
        if data_path is not None:
            self.analysis_folder = data_path
        elif self.analysis_folder is None:
            self.analysis_folder = self.data_folder
    
        with open(self.analysis_folder.joinpath("Parameters.yml"), "w") as file:
            dict_to_save = dataclasses.asdict(self)
            dict_to_save['data_folder'] = dict_to_save['data_folder'].as_posix()
            dict_to_save['analysis_folder'] = dict_to_save['analysis_folder'].as_posix()
            yaml.dump(dict_to_save, file)
