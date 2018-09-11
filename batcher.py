import numpy as np
from random import shuffle


class Batch():
    def __init__(self, batch_size, value, drug, cell, positions):
        self.batch_size = batch_size
        self.positions = positions
        self.value = value
        self.drug = drug
        self.cell = cell
        self.offset = 0
        self.size = positions.shape[0]

    def mini_batch(self):
        if self.offset >= self.size:
            return None
        if self.offset + self.batch_size <= self.size:
            sub_posi = self.positions[self.offset : self.offset + self.batch_size]
        else:
            sub_posi = self.positions[self.offset : ]
        self.offset += self.batch_size
        cell = []
        drug = []
        value = []
        for row, col in sub_posi:
            drug.append(self.drug[row])
            cell.append(self.cell[col])
            value.append(self.value[row, col])
        return np.array(value), np.array(drug), np.array(cell)
    
    def whole_batch(self):
        cell = []
        drug = []
        value = []
        for row, col in self.positions:
            drug.append(self.drug[row])
            cell.append(self.cell[col])
            value.append(self.value[row, col])
        return np.array(value), np.array(drug), np.array(cell)
        
    def diy_batch(self, k):
        cell = []
        drug = []
        value = []
        for row, col in self.positions[range(k)]:
            drug.append(self.drug[row])
            cell.append(self.cell[col])
            value.append(self.value[row, col])
        return np.array(value), np.array(drug), np.array(cell)
    
    def reset(self):
        self.offset = 0

    def available(self):
        if self.offset < self.size:
            return True
        else:
            return False

drug_smile_dict = np.load("data/drug_onehot_smiles.npy", encoding="latin1").item()
drug_cell_dict = np.load("data/drug_cell_interaction.npy", encoding="latin1").item()
cell_mut_dict = np.load("data/cell_mut_matrix.npy", encoding="latin1").item()

c_chars = drug_smile_dict["c_chars"]
drug_names = drug_smile_dict["drug_names"]
drug_cids = drug_smile_dict["drug_cids"]
canonical = drug_smile_dict["canonical"]
canonical = np.transpose(canonical, (0, 2, 1))
cell_names = cell_mut_dict["cell_names"]
mut_names = cell_mut_dict["mut_names"]
cell_mut = cell_mut_dict["cell_mut"]

all_positions = drug_cell_dict["positions"]
np.random.shuffle(all_positions)
        
def screen_max_conc():
    max_conc = drug_cell_dict["Max_conc"]
    row_ic50 = drug_cell_dict["raw_ic50"]
    max_conc_positions = []
    for row, col in all_positions:
        if row_ic50[row, col] <= max_conc[row, col]:
            max_conc_positions.append([row, col])
    return np.array(max_conc_positions)

def load_data(batch_size, label_list, positions=all_positions):
    size = positions.shape[0]
    len1 = int(size * 0.8)
    len2 = int(size * 0.9)
    
    train_pos = positions[0 : len1]
    valid_pos = positions[len1 : len2]
    test_pos = positions[len2 : ]

    value_shape = drug_cell_dict["IC50"].shape
    value = np.zeros((value_shape[0], value_shape[1], len(label_list)))

    for i in range(len(label_list)):
        value[ :, :, i ] = drug_cell_dict[label_list[i]]
    drug_smile = canonical

    train = Batch(batch_size, value, drug_smile, cell_mut, train_pos)
    valid = Batch(batch_size, value, drug_smile, cell_mut, valid_pos)
    test = Batch(batch_size, value, drug_smile, cell_mut, test_pos)
    return train, valid, test

def load_max_conc_data(batch_size, label_list):
    max_conc_positions = screen_max_conc()
    return load_data(batch_size, label_list, positions=max_conc_positions)

def load_partial_data(batch_size, label_list, ratio, positions=all_positions):
    size = positions.shape[0]
    len1 = int(size * ratio)
    len2 = int(size * (ratio * 1.1))
    len3 = int(size * (ratio * 1.2))

    train_pos = positions[0 : len1]
    valid_pos = positions[len1 : len2]
    test_pos = positions[len2 : len3]

    value_shape = drug_cell_dict["IC50"].shape
    value = np.zeros((value_shape[0], value_shape[1], len(label_list)))

    for i in range(len(label_list)):
        value[ :, :, i ] = drug_cell_dict[label_list[i]]
    drug_smile = canonical

    train = Batch(batch_size, value, drug_smile, cell_mut, train_pos)
    valid = Batch(batch_size, value, drug_smile, cell_mut, valid_pos)
    test = Batch(batch_size, value, drug_smile, cell_mut, test_pos)
    return train, valid, test

def load_max_conc_partial_data(batch_size, label_list, ratio):
    max_conc_positions = screen_max_conc()
    return load_partial_data(batch_size, label_list, ratio)

def load_empty_data(batch_size, label_list, positions=all_positions):
    value_shape = drug_cell_dict["IC50"].shape
    value = np.zeros((value_shape[0], value_shape[1], len(label_list)))
    for i in range(len(label_list)):
        value[ :, :, i ] = drug_cell_dict[label_list[i]]

    existance = np.zeros((value_shape[0], value_shape[1]))
    for item in positions:
        existance[item[0], item[1]] = 1

    empty_row, empty_col = np.where(existance == 0)
    empty_pos = np.array(zip(empty_row, empty_col))
    drug_smile = canonical
    
    empty = Batch(batch_size, value, drug_smile, cell_mut, empty_pos)
    return empty

def load_drug_blind_data(batch_size, label_list, positions=all_positions):
    value_shape = drug_cell_dict["IC50"].shape
    value = np.zeros((value_shape[0], value_shape[1], len(label_list)))
    for i in range(len(label_list)):
        value[ :, :, i ] = drug_cell_dict[label_list[i]]
    
    drug_array = np.array(range(canonical.shape[0]))
    np.random.shuffle(drug_array)
    test_drug_number = int(drug_array.shape[0] * 0.1)
    other_drug_id = list(drug_array[test_drug_number :])
    test_drug_id = list(drug_array[0 : test_drug_number])
    
    other_pos = []
    test_pos = []
    for i in range(positions.shape[0]):
        if positions[i, 0] in other_drug_id:
            other_pos.append(positions[i])
            continue
        if positions[i, 0] in test_drug_id:
            test_pos.append(positions[i])
            continue
        print("warning") 
    other_len = len(other_pos)
    train_pos = np.array(other_pos[0 : int(other_len * 0.9) ])
    valid_pos = np.array(other_pos[int(other_len * 0.9) : ])
    test_pos = np.array(test_pos)
    
    drug_smile = canonical

    train = Batch(batch_size, value, drug_smile, cell_mut, train_pos)
    valid = Batch(batch_size, value, drug_smile, cell_mut, valid_pos)
    test = Batch(batch_size, value, drug_smile, cell_mut, test_pos)
    return train, valid, test


def load_cell_blind_data(batch_size, label_list, positions=all_positions):
    value_shape = drug_cell_dict["IC50"].shape
    value = np.zeros((value_shape[0], value_shape[1], len(label_list)))
    
    cell_array = np.array(range(cell_names.shape[0]))
    np.random.shuffle(cell_array)
    test_drug_number = int(cell_array.shape[0] * 0.1)
    test_cell_id = list(cell_array[0 : test_drug_number])
    other_cell_id = list(cell_array[test_drug_number : ])

    positions = drug_cell_dict["positions"]
    
    other_pos = []
    test_pos = []
    for i in range(positions.shape[0]):
        if positions[i, 1] in other_cell_id:
            other_pos.append(positions[i])
            continue
        if positions[i, 1] in test_cell_id:
            test_pos.append(positions[i])
            continue
        print("warning")
    other_len = len(other_pos)
    train_pos = np.array(other_pos[0 : int(other_len * 0.9)])
    valid_pos = np.array(other_pos[int(other_len * 0.9): ])
    test_pos = np.array(test_pos)
    
    for i in range(len(label_list)):
        value[ :, :, i ] = drug_cell_dict[label_list[i]]
    drug_smile = canonical
    
    train = Batch(batch_size, value, drug_smile, cell_mut, train_pos)
    valid = Batch(batch_size, value, drug_smile, cell_mut, valid_pos)
    test = Batch(batch_size, value, drug_smile, cell_mut, test_pos)
    return train, valid, test

def load_tissue_data(batch_size, label_list, tissue_ids, positions=all_positions):
    tissue_one = cell_mut_dict["desc1"]
    tissue_two = cell_mut_dict["desc2"]

    tissue_names = list(set(list(tissue_one)))
    tissue_names = np.sort(tissue_names)
    tissue_dict = {}
    for i in range(len(tissue_names)):
        tissue_dict[tissue_names[i]] = i

    positions = drug_cell_dict["positions"]
    
    tissue_pos = []
    other_tissue_pos = []
    for i in range(positions.shape[0]):
        row = positions[i, 0]
        col = positions[i, 1]
        tissue = tissue_one[col]
        ids = tissue_dict[tissue]
        if ids == tissue_ids:
            tissue_pos.append(positions[i])
        else:
            other_tissue_pos.append(positions[i])
    
    other_tissue_pos = np.array(other_tissue_pos)
    np.random.shuffle(other_tissue_pos)
    size = other_tissue_pos.shape[0]
    len1 = int(size * 0.9)
    
    train_pos = other_tissue_pos[0 : len1]
    valid_pos = other_tissue_pos[len1 : ]
    test_pos = np.array(tissue_pos)

    value_shape = drug_cell_dict["IC50"].shape
    value = np.zeros((value_shape[0], value_shape[1], len(label_list)))

    for i in range(len(label_list)):
        value[ :, :, i ] = drug_cell_dict[label_list[i]]
    drug_smile = canonical

    train = Batch(batch_size, value, drug_smile, cell_mut, train_pos)
    valid = Batch(batch_size, value, drug_smile, cell_mut, valid_pos)
    test = Batch(batch_size, value, drug_smile, cell_mut, test_pos)
    
    return train, valid, test

def load_padel_drug_data(batch_size, label_list, positions=all_positions):
    drug_padel_dict = np.load("comparison_plos_one_2013/data/drug_padel_features.npy", encoding="latin1").item()
    drug_names = drug_padel_dict["drug_names"]
    drug_cids = drug_padel_dict["drug_cids"]
    canonical = drug_padel_dict["canonical"]
    isomerics = drug_padel_dict["isomerics"]

    size = positions.shape[0]
    len1 = int(size * 0.8)
    len2 = int(size * 0.9)
    
    train_pos = positions[0 : len1]
    valid_pos = positions[len1 : len2]
    test_pos = positions[len2 : ]

    value_shape = drug_cell_dict["IC50"].shape
    value = np.zeros((value_shape[0], value_shape[1], len(label_list)))
    for i in range(len(label_list)):
        value[ :, :, i ] = drug_cell_dict[label_list[i]]
    drug_smile = canonical

    train = Batch(batch_size, value, drug_smile, cell_mut, train_pos)
    valid = Batch(batch_size, value, drug_smile, cell_mut, valid_pos)
    test = Batch(batch_size, value, drug_smile, cell_mut, test_pos)
    return train, valid, test

def load_part_feature_data(batch_size, label_list, positions=all_positions, mut_size=310, cna_size=425):
    size = positions.shape[0]
    len1 = int(size * 0.8)
    len2 = int(size * 0.9)
    mut_list = []
    cna_list = []
    for i in range(len(mut_names)):
        name = mut_names[i]
        if name.startswith(b"cna"):
            cna_list.append(i)
        if name.endswith(b"mut"):
            mut_list.append(i)
    
    mut_choice = np.random.choice(mut_list, mut_size, replace=False)
    cna_choice = np.random.choice(cna_list, cna_size, replace=False)
    mut_choice.sort()
    cna_choice.sort()
    
    mask = np.concatenate([mut_choice, cna_choice])
    part_mut = cell_mut[:, mask]

    train_pos = positions[0 : len1]
    valid_pos = positions[len1 : len2]
    test_pos = positions[len2 : ]

    value_shape = drug_cell_dict["IC50"].shape
    value = np.zeros((value_shape[0], value_shape[1], len(label_list)))

    for i in range(len(label_list)):
        value[ :, :, i ] = drug_cell_dict[label_list[i]]
    drug_smile = canonical

    train = Batch(batch_size, value, drug_smile, part_mut, train_pos)
    valid = Batch(batch_size, value, drug_smile, part_mut, valid_pos)
    test = Batch(batch_size, value, drug_smile, part_mut, test_pos)
    return train, valid, test

