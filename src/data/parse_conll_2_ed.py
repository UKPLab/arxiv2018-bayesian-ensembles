'''
Jonas Pfeiffer, 2019.
'''

import os

'''
Hey Edwin,

I wrote a parser for your format.
It assumes the Connl format in which the first column has the token and all subsequent columns have labels.
I generate an int label for each label in the same ordering and 
the last column has a 1 for the first word in the document(or sentence) 
and a 0 for all subsequent tokens of that document:

So an example:

Token  labels_1 labels_2
Jonas  B N
is O V
cool  O A

Edwin  B N
is O V
too O ADV

Would transform to

Token  labels_1 labels_2 labels_1_pos  labels_2_pos is_start
Jonas  B N   0 0 1
is     O V   1 1 0
cool   O A   1 2 0
Edwin  B N   0 0 1
is     O V   1 1 0
too    O ADV 1 3 0
'''

def get_label(positional_dict, label):
    # removed the separate dictionaries for each column. We need all annotations to be mapped to a common set of IDs in
    # these experiments.
    if label not in positional_dict:
        positional_dict[label] = len(positional_dict)
        print('Adding a new label: %s' % label)
    return positional_dict[label]


def load_file(file_name, positional_dict=None):

    if positional_dict is None:
        positional_dict = {}

    new_line = True

    new_file_name = file_name[:-4] + '_ES.tsv'  # changed the filenames for generated files

    with open(new_file_name, 'w', encoding='utf-8') as o:

        with open(file_name, 'r', encoding='utf-8') as f:

            for line in f:

                line.replace('\n', '')

                elems = line.split()

                if len(elems) == 0:
                    new_line = True
                    continue

                length = len(elems)

                for i in range(1, length):

                    elems.append(str(get_label(positional_dict, elems[i])))

                if new_line:
                    elems.append('1')
                else:
                    elems.append('0')

                o.write('\t'.join(elems) + '\n')

                new_line = False

    return positional_dict

rootdir = os.path.expanduser('./data/famulus/data/famulus_MEd') # TEd')

positional_dict = {
    'O': 1,
    'I_HG': 0,
    'B_HG': 2,
    'I_EG': 3,
    'B_EG': 4,
    'I_EE': 5,
    'B_EE': 6,
    'I_DC': 7,
    'B_DC': 8,
    'I-HypothesenGeneration': 0,
    'B-HypothesenGeneration': 2,
    'I-EvidenzGeneration': 3,
    'B-EvidenzGeneration': 4,
    'I-EvidenzEvaluation': 5,
    'B-EvidenzEvaluation': 6,
    'I-Schlussfolgerung': 7,
    'B-Schlussfolgerung': 8
}

# positional_dict = {}

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".txt"):
            positional_dict = load_file(filepath, positional_dict)
