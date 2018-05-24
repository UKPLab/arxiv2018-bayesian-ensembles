from argparse import ArgumentParser
import numpy as np
import pandas as pd

def get_fraction_matches(annos, annos_to_match_with, doc_starts, strict):

    annos = annos.flatten()
    annos_to_match_with = annos_to_match_with.flatten()

    valididxs = (annos != -1) & (annos_to_match_with != -1)
    annos[valididxs]
    annos_to_match_with[valididxs]

    B_labels = [2, 4, 6, 8]
    I_labels = [0, 3, 5, 7]

    prev = 1
    prevtype = -1
    nmatches = 0
    labtype = -1

    span_count = 0
    total_matches = 0

    for i, label in enumerate(annos):

        if label == 1 and prev != 1:
            # end of previous label, if there was one

            end = i
            length = end - start
            span_count += 1
            if strict:
                total_matches += nmatches == length
            else:
                total_matches += nmatches / float(length)

            prev = 1
            prevtype = -1
            nmatches = 0
            continue
        elif label == 1:
            continue

        # label type
        if label in I_labels:
            labtype = np.argwhere(I_labels == label)[0][0]
        elif label in B_labels:
            labtype = np.argwhere(B_labels == label)[0][0]

        if prevtype == -1 or doc_starts[i]:
            # new span
            start = i
            # reset matchs
            nmatches = 0
        elif labtype != prevtype or label in B_labels:
            # There was an old span. We will start a new one because the type has changed or there is a B label.
            # Count up for the old span
            end = i
            length = end - start
            span_count += 1
            if strict:
                total_matches += nmatches == length
            else:
                total_matches += nmatches / float(length)

            # new span
            start = i
            # reset matchs
            nmatches = 0

        # check match with targets
        if annos_to_match_with[i] in I_labels:
            target_labtype = np.argwhere(I_labels == annos_to_match_with[i])[0][0]
        elif annos_to_match_with[i] in B_labels:
            target_labtype = np.argwhere(B_labels == annos_to_match_with[i])[0][0]
        else:
            target_labtype = -1

        nmatches += labtype == target_labtype

        prev = label
        prevtype = labtype

    # finish any span that ran until the end of the file
    if prev != 1:
        # end of span
        end = i + 1
        length = end - start
        span_count += 1
        if strict:
            total_matches += nmatches == length
        else:
            total_matches += nmatches / float(length)

    frac_matches = total_matches / float(span_count)

    return frac_matches

def precision(predictions, gold, strict, doc_starts):
    return get_fraction_matches(annos=predictions, annos_to_match_with=gold, strict=strict, doc_starts=doc_starts)

def recall(predictions, gold, strict, doc_starts):
    return get_fraction_matches(annos=gold, annos_to_match_with=predictions, strict=strict, doc_starts=doc_starts)

def f1(predictions, gold, strict, doc_starts):
    prec = precision(predictions, gold, strict, doc_starts=doc_starts)
    rec = recall(predictions, gold, strict, doc_starts=doc_starts)
    return 2 * prec * rec / (prec + rec)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("goldfile", help="location of the gold labels")
    parser.add_argument("aggfile", help="location of the aggregated predictions")
    parser.add_argument('strict',
                        help="Can be 'strict' to count only matching spans, or 'relaxed' to count fractions of matching spans.")
    parser.add_argument('doc_starts',
                        help="location of a file containing ones to indicate where each sequence (sentence or document) starts.")

    args = parser.parse_args()

    gold = np.genfromtxt(args.goldfile, dtype=int)
    agg = pd.read_csv(args.aggfile)
    doc_starts = pd.read_csv(args.doc_starts).values

    if agg.shape[0] > gold.shape[0]:
        print('Truncating the predictions... assuming the first part matches the gold.')
        agg = agg.iloc[:gold.shape[0]]

    print('Predictions:')
    print(agg)
    print('Gold:')
    print(gold)
    print('doc_starts:')
    print(doc_starts)

    if agg.ndim == 1:
        agg = agg[:, None]

    for col in agg.columns:

        prec = get_fraction_matches(annos=agg[col].values, annos_to_match_with=gold, strict=args.strict=='strict', doc_starts=doc_starts)
        rec = get_fraction_matches(annos=gold, annos_to_match_with=agg[col].values, strict=args.strict=='strict', doc_starts=doc_starts)

        f1 = 2 * prec * rec / (prec + rec)

        print('Results for %s' % col)
        print('precision = %f' % prec)
        print('recall = %f' % rec)
        print('f1 = %f' % f1)