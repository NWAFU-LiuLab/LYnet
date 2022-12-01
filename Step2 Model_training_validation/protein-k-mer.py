# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 00:18:24 2022

@author: lvyang
"""

import numpy as np
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import tqdm
import os
import itertools
import work_with_files


def occurrences_count(string, sub):
    """
        Counting all ocurrances of substring in string using find() method

    Args:
        string: str, string where to find
        sub: str, string to find

    Returns:
        int, number of occurrances
    """
    # Starting index and count num
    count = start = 0

    # Counting
    while True:
        # If found count = count + 1
        # start = {where was found} +1

        start = string.find(sub, start) + 1
        if start > 0:
            count += 1
        else:
            return count


def making_aa_k_mers(k):
    """
    Making all possible subsequences with length k using aminoacids (order is important)
    Args:
        k: int, length of k-mer

    Returns:
        list of all possible aminoacid k-mer sequences
    """
    amino_string = 'ARNDCQEGHILKMFPSTWYV'

    # making all possible substrings
    subseq_iter = itertools.product(amino_string, repeat=k)
    aa_k_mer_list = list(subseq_iter)

    # one "for" to deal with tuples which we get from itertools stuff
    for i in range(len(aa_k_mer_list)):

        # tuples are in list
        tup = aa_k_mer_list[i]

        # rewriting list elements
        aa_k_mer_list[i] = ''.join(tup)

    return aa_k_mer_list


def seqio_data(seq_record):
    """
    Working with SeqRecord class

    Args:
        seq_record: SeqRecord class from Biopython

    Returns:
        protein_name: str, name of protein
        sequence: str, sequence of protein
    """

    # getting protein name
    protein = seq_record.name

    # getting sequence
    seq = str(seq_record.seq)

    return protein, seq


def finding_freq_single_protein(seq, aa_k_mer_list):
    """
    Finding frequnces for subsequences in single protein
    and scaling it with SKlearn StandardScaler()

    Args:
        seq: str, sequence of amino acids in protein
        aa_k_mer_list: lst, all possible k-mers for aminoacids

    Returns:
        list, frequency of all k-mers from aa_k_mer_list,  vector is normalized using sklearn
    """

    # Getting initial sizes
    n = len(seq)

    # Initializing list where all frequencies will be saved
    vector_freq = []

    # Counting frequencies
    for x in aa_k_mer_list:
        vector_freq.append(float(occurrences_count(seq, x)) / n)

    # Making some prep with array
    vector_freq = np.array(vector_freq)
    vector_freq = vector_freq.reshape((-1, 1))

    # Standardizing our frequencies
    scaler = preprocessing.StandardScaler()
    vector_freq_scaled = scaler.fit_transform(vector_freq)
    
    # Getting return converting dimensions
    result = vector_freq_scaled.reshape(1, -1)[0].tolist()
    
    return result

def main_analysis(path, k_mer_num,output_file_name, trembl_usage_human=False,):
    """
    Construct "organism_name".csv with k-mer analyzes. Will store analyzed file in 'data/csv_data' directory.
    !!!Warning!!! can take much time, so be prepared and sure that all parameters are good
    Args:
        path: str, path to file in fasta format with represantative proteome used in analyzes
        k_mer_num: int, k-mer length
        trembl_usage: bool, default is False. Do you use represantative proteome for human with TrEMBL proteins or not?
    """
    # creating dir to store CSVs produced by function

    # initializing aa_subseqs
    aa_k_mer_list = making_aa_k_mers(k_mer_num)

    # initializing DataFrame
    table_columns = ['Organism', 'Protein'] + aa_k_mer_list
    proteins_data = pd.DataFrame(columns=table_columns)

    # reading
    prot_records= work_with_files.read_fasta(path)
    
    num_iter=1
    # dealing with human, because it needs to be analyzed separately
    if 1:

        # initializing list
        human_list = []

        # appending all human proteins to list and splitting it into 100 parts
        prot_records_split = np.array_split(prot_records, num_iter)
        for prot_data_part in prot_records_split:
            human_list.append(prot_data_part)

        # We will split analyze of human, because human proteom is too big to handle
        for j in tqdm.tqdm(range(0, num_iter)):

            # Creating pd.df
            proteins_data = pd.DataFrame(columns=table_columns)
            index = 0


            for i in range(len(human_list[j])):

                # taking exact protein and calculating metrics (frequencies)
                SeqRecord = human_list[j][i]
                prot_name, seq = seqio_data(SeqRecord)
                freq_vector = finding_freq_single_protein(seq, aa_k_mer_list)

                # making row for table
                adding_row = []
                adding_row.append(output_file_name)
                adding_row.append(prot_name)
                adding_row += freq_vector

                # adding row to the DataFrame
                proteins_data.loc[index] = adding_row
                index += 1

            # Writing file for every part of data, we will combine them later
            writing_path = "data2/"+output_file_name + f'_{ k_mer_num}_mer_' + '.csv'
            proteins_data.to_csv(writing_path)
    else:
        
        # Rewriting index
        index = 0

        # working with NOT human proteomes
        for i in tqdm.tqdm(range(len(prot_records))):

            # reading protein to calculate metrics on it
            seq_record = prot_records[i]
            prot_name, seq = seqio_data(seq_record)

            # calculating metrics (frequencies)
            freq_vector = finding_freq_single_protein(seq, aa_k_mer_list)

            # making row for pandas
            adding_row = []
            adding_row.append(output_file_name)
            adding_row.append(prot_name)
            adding_row += freq_vector

            # adding row to the DataFrame
            proteins_data.loc[index] = adding_row
            index += 1

        # Writing file
        writing_path = "data2/"+output_file_name + '_' + '.csv'
        proteins_data.to_csv(writing_path)


def main(file, k,data_type):
        main_analysis(file, k,data_type)
        
k_mer=2
#feature_number=20**k_mer
for i in["Train_Positive","Train_Negative","Test_Positive","Test_negative"]:        
    main(r"data2/%s.fasta"%i,k_mer,i)
