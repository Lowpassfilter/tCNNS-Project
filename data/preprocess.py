import os
import csv
from pubchempy import *
import numpy as np
import numbers
import h5py
import math




def is_not_float(string_list):
    try:
        for string in string_list:
            float(string)
        return False
    except:
        return True

"""
The following 4 function is used to preprocess the drug data. We download the drug list manually, and download the SMILES format using pubchempy. Since this part is time consuming, I write the cids and SMILES into a csv file. 
"""

#folder = "data/"
folder = ""

def load_drug_list():
    filename = folder + "Drug_listTue Oct 31 03_03_04 2017.csv"
    csvfile = open(filename, "rb")
    reader = csv.reader(csvfile)
    next(reader, None)
    drugs = []
    for line in reader:
        drugs.append(line[0])
    drugs = list(set(drugs))
    return drugs

def write_drug_cid():
    drugs = load_drug_list()
    drug_id = []
    datas = []
    outputfile = open(folder + 'pychem_cid.csv', 'wb')
    wr = csv.writer(outputfile)
    unknow_drug = []
    for drug in drugs:
        c = get_compounds(drug, 'name')
        if drug.isdigit():
            cid = int(drug)
        elif len(c) == 0:
            unknow_drug.append(drug)
            continue
        else:
            cid = c[0].cid
        print drug, cid
        drug_id.append(cid)
        row = [drug, str(cid)]
        wr.writerow(row)
    outputfile.close()
    outputfile = open(folder + "unknow_drug_by_pychem.csv", 'wb')
    wr = csv.writer(outputfile)
    wr.writerow(unknow_drug)

def cid_from_other_source():
    """
    some drug can not be found in pychem, so I try to find some cid manually.
    the small_molecule.csv is downloaded from http://lincs.hms.harvard.edu/db/sm/
    """
    f = open(folder + "small_molecule.csv", 'r')
    reader = csv.reader(f)
    reader.next()
    cid_dict = {}
    for item in reader:
        name = item[1]
        cid = item[4]
        if not name in cid_dict: 
            cid_dict[name] = str(cid)

    unknow_drug = open(folder + "unknow_drug_by_pychem.csv").readline().split(",")
    drug_cid_dict = {k:v for k,v in cid_dict.iteritems() if k in unknow_drug and not is_not_float([v])}
    return drug_cid_dict

def load_cid_dict():
    reader = csv.reader(open(folder + "pychem_cid.csv"))
    pychem_dict = {}
    for item in reader:
        pychem_dict[item[0]] = item[1]
    pychem_dict.update(cid_from_other_source())
    return pychem_dict

def download_smiles():
    cids_dict = load_cid_dict()
    cids = [v for k,v in cids_dict.iteritems()]
    inv_cids_dict = {v:k for k,v in cids_dict.iteritems()}
    download('CSV', folder + 'drug_smiles.csv', cids, operation='property/CanonicalSMILES,IsomericSMILES', overwrite=True)
    f = open(folder + 'drug_smiles.csv')
    reader = csv.reader(f)
    header = ['name'] + reader.next()
    content = []
    for line in reader:
        content.append([inv_cids_dict[line[0]]] + line)
    f.close()
    f = open(folder + "drug_smiles.csv", "w")
    writer = csv.writer(f)
    writer.writerow(header)
    for item in content:
        writer.writerow(item)
    f.close()

"""
The following code will convert the SMILES format into onehot format
"""
def string2smiles_list(string):
    char_list = []
    i = 1
    while i < len(string):
        c = string[i]
        if c.islower():
            char_list.append(string[i-1:i+1])
            i += 1
        else:
            char_list.append(string[i-1])
        i += 1
    if not string[-1].islower():
        char_list.append(string[-1])
    return char_list

def onehot_encode(char_list, smiles_string, length):
    encode_row = lambda char: map(int, [c == char for c in smiles_string])
    ans = np.array(map(encode_row, char_list))
    if ans.shape[1] < length:
        residual = np.zeros((len(char_list), length - ans.shape[1]), dtype=np.int8)
        ans = np.concatenate((ans, residual), axis=1)
    return ans

def smiles_to_onehot(smiles, c_chars, c_length):
    c_ndarray = np.ndarray(shape=(len(smiles), len(c_chars), c_length), dtype=np.float32)
    for i in range(len(smiles)):
        c_ndarray[i, ...] = onehot_encode(c_chars, smiles[i], c_length)
    return c_ndarray

def load_as_ndarray():
    reader = csv.reader(open(folder + "drug_smiles.csv"))
    next(reader, None)
    smiles = np.array(list(reader), dtype=np.str)
    return smiles

def charsets(smiles):
    union = lambda x, y: set(x) | set(y)
    c_chars = list(reduce(union, map(string2smiles_list, list(smiles[:, 2]))))
    i_chars = list(reduce(union, map(string2smiles_list, list(smiles[:, 3]))))
    return c_chars, i_chars

def save_drug_smiles_onehot():
    smiles = load_as_ndarray()
    # we will abandon isomerics smiles from now on
    c_chars, _ = charsets(smiles)
    c_length = max(map(len, map(string2smiles_list, list(smiles[:, 2]))))
    
    count = smiles.shape[0]
    drug_names = smiles[:, 0].astype(str)
    drug_cids = smiles[:, 1].astype(int)
    smiles = [string2smiles_list(smiles[i, 2]) for i in range(count)]
    
    canonical = smiles_to_onehot(smiles, c_chars, c_length)
    
    save_dict = {}
    save_dict["drug_names"] = drug_names
    save_dict["drug_cids"] = drug_cids
    save_dict["canonical"] = canonical
    save_dict["c_chars"] = c_chars

    print "drug onehot smiles data:"
    print drug_names.shape
    print drug_cids.shape
    print canonical.shape

    np.save(folder + "drug_onehot_smiles.npy", save_dict)
    print "finish saving drug onehot smiles data:"
    return drug_names, drug_cids, canonical

"""
The following part will prepare the mutation features for the cell.
"""

def save_cell_mut_matrix():
    f = open(folder + "PANCANCER_Genetic_feature_Tue Oct 31 03_00_35 2017.csv")
    reader = csv.reader(f)
    reader.next()
    cell_dict = {}
    mut_dict = {}

    matrix_list = []
    organ1_dict = {}
    organ2_dict = {}
    for item in reader:
        cell = item[0]
        mut = item[5]
        organ1_dict[cell] = item[2]
        organ2_dict[cell] = item[3]
        is_mutated = int(item[6])
        if cell in cell_dict:
            row = cell_dict[cell]
        else:
            row = len(cell_dict)
            cell_dict[cell] = row
        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col
        matrix_list.append((row, col, is_mutated))
    
    matrix = np.ones(shape=(len(cell_dict), len(mut_dict)), dtype=np.float32)
    matrix = matrix * -1
    for item in matrix_list:
        matrix[item[0], item[1]] = item[2]

    feature_num = [len(filter(lambda x: x >=0, list(matrix[i, :]))) for i in range(len(cell_dict))]
    indics = [i for i in range(len(feature_num)) if feature_num[i]==735]
    matrix = matrix[indics, :]

    inv_cell_dict = {v:k for k,v in cell_dict.iteritems()}
    all_names = [inv_cell_dict[i] for i in range(len(inv_cell_dict))]
    cell_names = np.array([all_names[i] for i in indics])

    inv_mut_dict = {v:k for k,v in mut_dict.iteritems()}
    mut_names = np.array([inv_mut_dict[i] for i in range(len(inv_mut_dict))])
    
    desc1 = []
    desc2 = []
    for i in range(cell_names.shape[0]):
        desc1.append(organ1_dict[cell_names[i]])
        desc2.append(organ2_dict[cell_names[i]])
    desc1 = np.array(desc1)
    desc2 = np.array(desc2)

    save_dict = {}
    save_dict["cell_mut"] = matrix
    save_dict["cell_names"] = cell_names
    save_dict["mut_names"] = mut_names
    save_dict["desc1"] = desc1
    save_dict["desc2"] = desc2

    print "cell mut data:"
    print len(all_names)    
    print cell_names.shape
    print mut_names.shape
    print matrix.shape
    np.save(folder + "cell_mut_matrix.npy", save_dict)
    print "finish saving cell mut data:"

    return matrix, cell_names, mut_names

#save_cell_mut_matrix()
#exit()
"""
This part is used to extract the drug - cell interaction strength. it contains IC50, AUC, Max conc, RMSE, Z_score

"""

def save_drug_cell_matrix():
    f = open(folder + "PANCANCER_IC_Tue Oct 31 02_59_53 2017.csv")
    reader = csv.reader(f)
    reader.next()

    drug_dict = {}
    cell_dict = {}
    matrix_list = []

    for item in reader:
        drug = item[0]
        cell = item[2]
        
        if drug in drug_dict:
            row = drug_dict[drug]
        else:
            row = len(drug_dict)
            drug_dict[drug] = row
        if cell in cell_dict:
            col = cell_dict[cell]
        else:
            col = len(cell_dict)
            cell_dict[cell] = col
        
        matrix_list.append((row, col, item[8], item[9], item[10], item[11], item[12]))
        
    existance = np.zeros(shape=(len(drug_dict), len(cell_dict)), dtype=np.int32)
    matrix = np.zeros(shape=(len(drug_dict), len(cell_dict), 6), dtype=np.float32)
    for item in matrix_list:
        existance[item[0], item[1]] = 1
        matrix[item[0], item[1], 0] = 1 / (1 + pow(math.exp(float(item[2])), -0.1))
        matrix[item[0], item[1], 1] = float(item[3])
        matrix[item[0], item[1], 2] = float(item[4])
        matrix[item[0], item[1], 3] = float(item[5])
        matrix[item[0], item[1], 4] = float(item[6])
        matrix[item[0], item[1], 5] = math.exp(float(item[2]))

    
    inv_drug_dict = {v:k for k,v in drug_dict.iteritems()}
    inv_cell_dict = {v:k for k,v in cell_dict.iteritems()}
    
    drug_names, drug_cids, canonical = save_drug_smiles_onehot()
    cell_mut_matrix, cell_names, mut_names = save_cell_mut_matrix()
    
    drug_ids = [drug_dict[i] for i in drug_names]
    cell_ids = [cell_dict[i] for i in cell_names]
    sub_matrix = matrix[drug_ids, :][:, cell_ids]
    existance = existance[drug_ids, :][:, cell_ids]
    
    row, col = np.where(existance > 0)
    positions = np.array(zip(row, col))
   
    print positions.shape

    save_dict = {}
    save_dict["drug_names"] = drug_names
    save_dict["cell_names"] = cell_names
    save_dict["positions"] = positions
    save_dict["IC50"] = sub_matrix[:, :, 0]
    save_dict["AUC"] = sub_matrix[:, :, 1]
    save_dict["Max_conc"] = sub_matrix[:, :, 2]
    save_dict["RMSE"] = sub_matrix[:, :, 3]
    save_dict["Z_score"] = sub_matrix[:, :, 4]
    save_dict["raw_ic50"] = sub_matrix[:, :, 5]

    print "drug cell interaction data:"
    print drug_names.shape
    print cell_names.shape
    print sub_matrix.shape
    print matrix.shape
   
    np.save(folder + "drug_cell_interaction.npy", save_dict)
    print "finish saving drug cell interaction data:"
    return sub_matrix
    
save_drug_cell_matrix()
