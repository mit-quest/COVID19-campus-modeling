import copy
import csv
from collections import namedtuple
from pathlib import Path


class MITBuildings:
    _buildings = dict()
    _possible_bldg_names = dict()
    DATA_FOLDER_FPATH = Path('models', 'common', 'data')

    def __init__(self, data_folder_fpath='') -> None:
        if data_folder_fpath:
            self.data_folder_fpath = data_folder_fpath
        else:
            repo_path = Path(__file__).absolute().parents[2]
            self.data_folder_fpath = Path(repo_path, self.DATA_FOLDER_FPATH)
        self.building_list_fpath = Path(
            self.data_folder_fpath, 'mit_building_list_all.csv')
        self._import_building_list()

    def known_building_ids(self) -> list:
        """
        Returns a list of all known MIT buildings, according to the MIT Facilities building list
        """
        building_ids = []
        for _, bldg_property in self._buildings.items():
            building_ids.append(bldg_property.building_num)
        return building_ids

    def is_valid_building_id(self, id: str) -> bool:
        """
        Returns if id exists in the list of known building IDs from various sources
        """
        if id.lower() in self._possible_bldg_names:
            return True
        return False

    def building_properties(self, id: str) -> namedtuple:
        """
        Obtains building properties from santized version of id
        """
        bldg_key = self._sanitize(id)
        return self._buildings[bldg_key]

    def pretty_name(self, id: str) -> str:
        """
        Returns human readable name of building ID from MIT Facilities building list
        """
        bldg_key = self._sanitize(id)
        return self._buildings[bldg_key].name

    def pretty_num(self, id: str) -> str:
        """
        Returns human readable building ID from MIT Facilities building list
        """
        bldg_key = self._sanitize(id)
        return self._buildings[bldg_key].building_num

    def _sanitize(self, s: str) -> str:
        """
        Private method: Tries to map the string s to an entry on the MIT Facilities building list
        Error is raised if is_valid_building_id(s) = False
        """
        if s.lower() in self._possible_bldg_names:
            return self._possible_bldg_names[s.lower()]
        raise ValueError('Unknown building name: ' + s)

    def _import_building_list(self):
        """
        Private method: Import building list from file and pre-process for easy retrieval
        """
        with open(self.building_list_fpath) as f:
            records = csv.DictReader(f)

            # Create Building tuple from headers
            headers = copy.deepcopy(records.fieldnames)
            headers.remove('bldg_wifi')
            headers.remove('bldg_pi_survey')
            headers.remove('bldg_provost')
            Building = namedtuple('Building', headers)

            for row in records:
                # Get building key
                row_dict = dict(row)
                bldg_key = row_dict['building_num'].lower()

                if not bldg_key:
                    continue

                # Add possible building names from MIT facilities list, wifi names, PI survey and provost list
                self._possible_bldg_names[bldg_key] = bldg_key

                wifi_names = row_dict['bldg_wifi'].lower().split('|')
                for wifi_name in wifi_names:
                    if wifi_name:
                        self._possible_bldg_names[wifi_name] = bldg_key

                pi_survey_names = row_dict['bldg_pi_survey'].lower().split('|')
                for pi_survey_name in pi_survey_names:
                    if pi_survey_name:
                        self._possible_bldg_names[pi_survey_name] = bldg_key

                provost_names = row_dict['bldg_provost'].lower().split('|')
                for provost_name in provost_names:
                    if provost_name:
                        self._possible_bldg_names[provost_name] = bldg_key

                # Add building entry
                row_dict.pop('bldg_wifi')
                row_dict.pop('bldg_pi_survey')
                row_dict.pop('bldg_provost')
                self._buildings[bldg_key] = Building(**row_dict)
