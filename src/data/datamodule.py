import pytorch_lightning as pl
import pandas as pd
import pickle
import numpy as np
from hydra.utils import to_absolute_path

# from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader


"""Helper class for specific training/testing dataset"""
class TrainingDataset(Dataset):
    def __init__(self, x0_values, x0_classes, x1_values, times_x0, times_x1):
        self.x0_values = x0_values
        self.x0_classes = x0_classes
        self.x1_values = x1_values
        self.times_x0 = times_x0
        self.times_x1 = times_x1

    def __len__(self):
        return len(self.x0_values)

    def __getitem__(self, idx):
        return (self.x0_values[idx], self.x0_classes[idx], self.x1_values[idx], self.times_x0[idx], self.times_x1[idx])

class PatientDataset(Dataset):
    def __init__(self, patient_data):
        self.patient_data = patient_data

    def __len__(self):
        return len(self.patient_data)

    def __getitem__(self, idx):
        return self.patient_data[idx]

class clinical_DataModule(pl.LightningDataModule):

    """returns: 
        x0_values, x0_classes, x1_values, times_x0, times_x1
    """

    def __init__(self, 
                 train_consecutive=False,
                 batch_size=256, 
                 file_path=None,
                 naming = None,
                 t_headings = None,
                 x_headings = [None],
                 cond_headings = [None],
                 memory=0):
        super().__init__()
        self.batch_size = batch_size
        self.file_path = file_path
        self.x_headings = x_headings
        self.cond_headings = cond_headings
        self.t_headings = t_headings 
        self.input_dim = len(self.x_headings) + len(self.cond_headings)
        self.output_dim = len(self.x_headings)
        self.naming = naming 
        self.memory = memory
        self.min_timept = 5 + self.memory
        self.train_consecutive = train_consecutive
        print("DataModule initialized to x_headings: ", self.x_headings, " cond_headings: ", self.cond_headings, " t_headings: ", self.t_headings, " train_consecutive: ", self.train_consecutive)

    def prepare_data(self):
        pass

    def __filter_data(self, data_set):
        # filter out data with less than 5 time points
        return data_set.groupby('HADM_ID').filter(lambda x: len(x) > self.min_timept)

    def __unpack__(self, data_set):
        x = data_set[self.x_headings].values
        cond = data_set[self.cond_headings].values
        t = data_set[self.t_headings].values
        return x, cond, t
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx].values.astype(np.float32)
        return sample
    
    def setup(self, stage=None):
        if not self.file_path:
            raise ValueError("data_module.file_path must be configured")

        data_path = to_absolute_path(self.file_path)
        self.data = pd.read_pickle(data_path)

        required_splits = {"train", "val", "test"}
        missing_splits = required_splits.difference(self.data.keys())
        if missing_splits:
            raise ValueError(
                f"Dataset {data_path} is missing required splits: "
                f"{sorted(missing_splits)}"
            )

        time_columns = ([self.t_headings] if isinstance(self.t_headings, str)
                        else list(self.t_headings))
        required_columns = {
            "HADM_ID", *self.x_headings, *self.cond_headings, *time_columns
        }
        for split_name in required_splits:
            missing_columns = required_columns.difference(
                self.data[split_name].columns
            )
            if missing_columns:
                raise ValueError(
                    f"Dataset split '{split_name}' is missing required columns: "
                    f"{sorted(missing_columns)}"
                )

        if not getattr(self, "_diagnostics_reported", False):
            self._report_numerical_diagnostics(data_path)
            self._diagnostics_reported = True

        if stage == 'fit' or stage is None:
            self.train = self.__filter_data(self.data['train'])
            self.val = self.__filter_data(self.data['val'])
        if stage == 'test' or stage is None:
            self.test = self.__filter_data(self.data['test'])

    def _report_numerical_diagnostics(self, data_path):
        """Print read-only dataset diagnostics without transforming the data."""
        print(f"[TFM-DIAG][data] source={data_path}")
        value_columns = list(self.x_headings) + list(self.cond_headings)
        time_column = self.t_headings

        for split_name in ("train", "val", "test"):
            split = self.data[split_name]
            diagnostic_columns = value_columns + [time_column]
            numeric = split[diagnostic_columns].apply(
                pd.to_numeric, errors="coerce"
            )
            values = numeric.to_numpy(dtype=np.float64)
            nan_count = int(np.isnan(values).sum())
            posinf_count = int(np.isposinf(values).sum())
            neginf_count = int(np.isneginf(values).sum())

            times = numeric[time_column].to_numpy(dtype=np.float64)
            finite_times = times[np.isfinite(times)]
            time_min = float(finite_times.min()) if finite_times.size else np.nan
            time_max = float(finite_times.max()) if finite_times.size else np.nan
            below_zero = int(np.sum(finite_times < 0))
            above_one = int(np.sum(finite_times > 1))

            deltas = []
            for _, group in split.groupby("HADM_ID"):
                patient_times = pd.to_numeric(
                    group[time_column], errors="coerce"
                ).sort_values().to_numpy(dtype=np.float64)
                if patient_times.size > 1:
                    deltas.append(np.diff(patient_times))
            deltas = np.concatenate(deltas) if deltas else np.array([])
            finite_deltas = deltas[np.isfinite(deltas)]
            delta_min = (float(finite_deltas.min())
                         if finite_deltas.size else np.nan)
            delta_max = (float(finite_deltas.max())
                         if finite_deltas.size else np.nan)
            negative_deltas = int(np.sum(finite_deltas < 0))
            zero_deltas = int(np.sum(finite_deltas == 0))
            invalid_deltas = int(np.sum(~np.isfinite(deltas)))

            print(
                f"[TFM-DIAG][data][{split_name}] rows={len(split)} "
                f"patients={split['HADM_ID'].nunique()} nan={nan_count} "
                f"+inf={posinf_count} -inf={neginf_count} "
                f"time_min={time_min:.8g} time_max={time_max:.8g} "
                f"time_lt_0={below_zero} time_gt_1={above_one} "
                f"dt_min={delta_min:.8g} dt_max={delta_max:.8g} "
                f"dt_lt_0={negative_deltas} dt_eq_0={zero_deltas} "
                f"dt_invalid={invalid_deltas}"
            )

            for column in diagnostic_columns:
                column_values = numeric[column].to_numpy(dtype=np.float64)
                invalid = int(np.sum(~np.isfinite(column_values)))
                if invalid:
                    print(
                        f"[TFM-DIAG][data][{split_name}] "
                        f"column={column} nonfinite={invalid}"
                    )

    def __sort_group__(self, data_set):
        grouped = data_set.groupby('HADM_ID')
        grouped_sorted = grouped.apply(lambda x: x.sort_values([self.t_headings], ascending = True)).reset_index(drop=True)
        return grouped_sorted

    def train_dataloader(self, shuffle=True):
        if not self.train_consecutive:
            train_data = self.create_pairs(self.train)
            # print(len(train_data))
            train_dataset = TrainingDataset(*train_data)
            return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=1)
        else:
            train_data = self.create_patient_data_t0(self.train)
            train_dataset = PatientDataset(train_data)
            return DataLoader(train_dataset, batch_size=1, shuffle=shuffle, num_workers=1)
    
    def val_dataloader(self):
        if self.train_consecutive:
            val_data = self.create_patient_data_t0(self.val)
            val_dataset = PatientDataset(val_data)
            return DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
        val_data = self.create_patient_data(self.val)
        val_dataset = PatientDataset(val_data)
        return DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    def test_dataloader(self):
        if self.train_consecutive:
            test_data = self.create_patient_data_t0(self.test)
            test_dataset = PatientDataset(test_data)
            return DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
        test_data = self.create_patient_data(self.test)
        test_dataset = PatientDataset(test_data)
        return DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    def create_patient_data(self, df):
        """Create formatted patient data from the DataFrame
        ex. patient 0: x0_values, x0_classes, times_x0...

        Args:
            df (_type_): _description_
            time_column (str, optional): _description_. Defaults to 'time_normalized'.
        """
        patient_lst = []
        for _, group in df.groupby('HADM_ID'):

            x0_values = []
            x0_classes = []
            x1_values = []
            times_x0 = []
            times_x1 = []

            sorted_group = group.sort_values(by=self.t_headings)
            x0_values, x0_classes, x1_values, times_x0, times_x1 = self.create_pairs(sorted_group)

            if len(self.cond_headings)<2:
                x0_classes = np.expand_dims(x0_classes, axis=1)
            else:
                x0_classes = x0_classes.squeeze().astype(np.float32)
            

            patient_lst.append((x0_values.squeeze().astype(np.float32), 
                                x0_classes, 
                                x1_values.squeeze().astype(np.float32), 
                                times_x0.squeeze().astype(np.float32), 
                                times_x1.squeeze().astype(np.float32)))
        return patient_lst

    
    def create_pairs(self, df):
        """create pairs of consecutive points from the DataFrame (for training the model)

        Args:
            df (pandas.DataFrame): _description_
            time_column (str, optional): _description_. Defaults to 'time_normalized'.

        Returns:
            numpy.array : x0_values, x0_classes, x1_values, times_x0, times_x1

        """
        # Initialize empty lists to store the components of the pairs
        x0_values = []
        x0_classes = []
        x1_values = []
        times_x0 = []
        times_x1 = []

        # Group the DataFrame by HADM_ID and iterate through each group
        for _, group in df.groupby('HADM_ID'):
            # Sort the group by time_normalized
            sorted_group = group.sort_values(by=self.t_headings)
        
            # Iterate through the sorted group to create pairs of consecutive points
            for i in range(self.memory,len(sorted_group) - 1):
                x0 = sorted_group.iloc[i]
                x0_class = x0[self.cond_headings].values
                x0_value = x0[self.x_headings].values

                x1 = sorted_group.iloc[i + 1]
                x1_value = x1[self.x_headings].values

                # memory component
                if self.memory>0:
                    x0_memory = sorted_group.iloc[i - self.memory:i]
                    x0_memory_flatten = x0_memory[self.x_headings].values.flatten()
                    x0_class = np.append(x0_class, x0_memory_flatten)

                x0_values.append(x0_value)
                x0_classes.append(x0_class)
                x1_values.append(x1_value)
                times_x0.append(x0[self.t_headings])
                times_x1.append(x1[self.t_headings])

        # Convert the lists to arrays
        x0_values = np.array(x0_values).squeeze().astype(np.float32)
        x0_classes = np.array(x0_classes).squeeze().astype(np.float32)
        x1_values = np.array(x1_values).squeeze().astype(np.float32)
        times_x0 = np.array(times_x0).squeeze().astype(np.float32)
        times_x1 = np.array(times_x1).squeeze().astype(np.float32)

        if len(self.cond_headings)<2:
                x0_classes = np.expand_dims(x0_classes, axis=1)

        return x0_values, x0_classes, x1_values, times_x0, times_x1


    def create_patient_data_t0(self, df):
        """Create formatted patient data from the DataFrame
        ex. patient 0: x0_values, x0_classes, times_x0...
        This version has x0 constant and x1 varying (as well as t)

        Args:
            df (_type_): _description_
            time_column (str, optional): _description_. Defaults to 'time_normalized'.
        """
        patient_lst = []
        for _, group in df.groupby('HADM_ID'):

            x0_values = []
            x0_classes = []
            x1_values = []
            times_x0 = []
            times_x1 = []

            sorted_group = group.sort_values(by=self.t_headings)
            x0_values, x0_classes, x1_values, times_x0, times_x1 = self.create_pairs(sorted_group)

            if len(self.cond_headings)<2:
                x0_classes = np.expand_dims(x0_classes, axis=1)
            else:
                x0_classes = x0_classes.squeeze().astype(np.float32) 

            # repeat the first point for x0_values and x0_classes, and times_x0
            x0_values = np.repeat(x0_values[0][None, :], len(x0_values), axis=0)
            x0_classes = np.repeat(x0_classes[0][None, :], len(x0_values), axis=0)
            times_x0 = np.repeat(times_x0[0], len(x0_values))

            patient_lst.append((x0_values.squeeze().astype(np.float32), 
                                x0_classes, 
                                x1_values.squeeze().astype(np.float32), 
                                times_x0.squeeze().astype(np.float32), 
                                times_x1.squeeze().astype(np.float32)))
        return patient_lst
    
    @property
    def dims(self):
        # x, cond, t
        return len(self.x_headings), len(self.cond_headings), len(self.t_headings)
