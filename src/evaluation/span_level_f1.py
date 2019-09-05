from argparse import ArgumentParser
import numpy as np
import pandas as pd

def get_fraction_matches(annos, annos_to_match_with, doc_starts, strict):

    annos = annos.flatten()
    annos_to_match_with = annos_to_match_with.flatten()

    valididxs = (annos != -1) & (annos_to_match_with != -1)
    annos = annos[valididxs]
    annos_to_match_with = annos_to_match_with[valididxs]

    B_labels = [2, 4, 6, 8]
    I_labels = [0, 3, 5, 7]

    prev = 1
    prevtype = -1
    nmatches = 0
    labtype = -1

    span_count = 0
    total_matches = 0

    def endspan(start, end, span_count, nmatches, total_matches, prevtype):
        length = end - start
        span_count += 1
        if strict:

            # check that the anno_to_match also ended at the same place
            if end < len(annos_to_match_with):

                matching = nmatches == length

                if annos_to_match_with[end] in I_labels and np.argwhere(I_labels == annos_to_match_with[i])[0][0] == prevtype:
                   # the span in annos_to_match_with has gone too far
                   matching = False
            else:
                matching = nmatches == length

            total_matches += matching
        else:
            total_matches += nmatches / float(length)
        return span_count, total_matches

    for i, label in enumerate(annos):

        if label == 1 and prev != 1:
            # end of previous label, if there was one

            span_count, total_matches = endspan(start, i, span_count, nmatches, total_matches, prevtype)

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
            span_count, total_matches = endspan(start, i, span_count, nmatches, total_matches, prevtype)

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

        if strict: # check that the start and end are the same location
            if label in B_labels and annos_to_match_with[i] in B_labels or \
                label in I_labels and annos_to_match_with[i] in I_labels:
                nmatches += labtype == target_labtype
        else:
            nmatches += labtype == target_labtype

        prev = label
        prevtype = labtype

    # finish any span that ran until the end of the file
    if prev != 1:
        # end of span
        span_count, total_matches = endspan(start, i+1, span_count, nmatches, total_matches, prevtype)

    if span_count == 0:
        frac_matches = 0
    else:
        frac_matches = total_matches / float(span_count)

    return frac_matches

def precision(predictions, gold, strict, doc_starts):
    return get_fraction_matches(annos=predictions, annos_to_match_with=gold, strict=strict, doc_starts=doc_starts)

def recall(predictions, gold, strict, doc_starts):
    return get_fraction_matches(annos=gold, annos_to_match_with=predictions, strict=strict, doc_starts=doc_starts)

def f1(predictions, gold, strict, doc_starts):
    prec = precision(predictions, gold, strict, doc_starts=doc_starts)
    rec = recall(predictions, gold, strict, doc_starts=doc_starts)

    if prec + rec == 0:
        #print('Warning: zero precision and recall scores!')
        return 0

    return 2 * prec * rec / (prec + rec)


def get_spans(labels, doc_starts):
    B_labels = [2, 4, 6, 8]
    I_labels = [0, 3, 5, 7]

    span_starts = []
    span_ends = []
    span_types = []
    prevtype = -1

    for i, tok in enumerate(labels):

        if tok in B_labels:
            labeltype = np.argwhere(B_labels == tok)[0][0]
        elif tok in I_labels:
            labeltype = np.argwhere(I_labels == tok)[0][0]
        else:
            labeltype = -1

        if (doc_starts[i] or tok in B_labels or labeltype == -1 or labeltype != prevtype) and len(span_starts) > len(span_ends):
            span_ends.append(i)

        if tok in B_labels:
            span_starts.append(i)
            span_types.append(labeltype)

        # # bad spans start with an I -- include them anyway?
        # ner metrics with this switched on:
        # 0.7238678090575276
        # 0.5469848316685164
        # 0.6231166368138237
        # Switched off:
        # 0.7494293685011413
        # 0.5492565055762082
        # 0.6339161214201438
        # elif tok in I_labels and len(span_starts) == len(span_ends):
        #     # we have an I label but no open span. Assume this is the start
        #     span_starts.append(i)
        #     span_types.append(labeltype)

        prevtype = labeltype

    if len(span_ends) < len(span_starts):
        span_ends.append(i + 1)

    spans = list(zip(span_starts, span_ends, span_types))

    return spans

def strict_span_metrics_2(predictions, gold, doc_starts):
    # retrieve spans (Start and end points)
    pspans = get_spans(predictions, doc_starts)
    gspans = get_spans(gold, doc_starts)

    # TP
    tp = np.sum([1 if span in gspans else 0 for span in pspans])

    # FP
    fp = len(pspans) - tp

    # FN
    fn = len(gspans) - tp

    if tp + fp == 0:
        prec = 0
    else:
        prec = tp / float(tp + fp)
    rec = tp / float(tp + fn)

    if prec + rec == 0:
        f1 = 0
    else:
        f1 = 2 * prec * rec / (prec + rec)

    return prec, rec, f1

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
    print(agg.shape)
    print('Gold:')
    print(gold.shape)
    print('doc_starts:')
    print(doc_starts.shape)

    if agg.ndim == 1:
        agg = agg[:, None]

    for col in agg.columns:

        prec = get_fraction_matches(annos=agg[col].values, annos_to_match_with=gold, strict=args.strict=='strict', doc_starts=doc_starts)
        rec = get_fraction_matches(annos=gold, annos_to_match_with=agg[col].values, strict=args.strict=='strict', doc_starts=doc_starts)

        if prec + rec > 0:
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = 0

        print('Results for %s' % col)
        print('precision = %f' % prec)
        print('recall = %f' % rec)
        print('f1 = %f' % f1)