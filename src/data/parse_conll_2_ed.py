import os

def get_label(positional_dict, position, label):
    if position not in positional_dict:
        positional_dict[position] = {}
    if label not in positional_dict[position]:
        positional_dict[position][label] = len(positional_dict[position])
    return positional_dict[position][label]


def load_file(file_name, positional_dict=None):

    if positional_dict is None:
        positional_dict = {}

    new_line = True

    with open(file_name + '_ed', 'w', encoding='utf-8') as o:

        with open(file_name, 'r', encoding='utf-8') as f:

            for line in f:

                line.replace('\n', '')

                elems = line.split()

                if len(elems) == 0:
                    new_line = True
                    continue


                length = len(elems)

                for i in range(1, length):

                    elems.append(str(get_label(positional_dict, i, elems[i])))

                if new_line:
                    elems.append('1')
                else:
                    elems.append('0')

                o.write('\t'.join(elems) + '\n')

                new_line = False

    return positional_dict

rootdir = 'data/'


for subdir, dirs, files in os.walk(rootdir):
    positional_dict = {}
    for file in files:
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file

        if filepath.endswith(".txt"):
            positional_dict = load_file(filepath, positional_dict)









