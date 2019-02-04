'''
Obtained from https://github.com/yinfeiy/PICO-data/

'''
import os
import json
import random
import spacy

random.seed(1395)
sp = spacy.load('en')

class Doc:

    def __init__(self, docid, markups_offset, spacydoc=None, genres=None):
        self.docid = docid
        self.spacydoc = spacydoc
        self.ntokens = len(spacydoc)

        self.markups_offset = markups_offset  ## offset markups on string character level
        self.markups = self.offset2markups(markups_offset)  ## markups on token level

        self.groundtruth_markups = None
        self.groundtruth_offset = None
        self.groundtruth = None

        self.genres = genres


    def offset2markups(self, markups_offset):
        markups = dict()

        offset2token_map = [0]*len(self.spacydoc.text)

        for i in range(self.ntokens):
            token = self.spacydoc[i]
            for j in range(len(token)):
                offset2token_map[token.idx + j] = i
        for i in range(1, len(offset2token_map)):
            offset2token_map[i] = max(offset2token_map[i], offset2token_map[i-1])

        for annotype in markups_offset:
            markups[annotype] = {}

            for wid in markups_offset[annotype]:
                markups[annotype][wid] = []

                for offset_span in markups_offset[annotype][wid]:
                    if offset_span[1] > len(offset2token_map): # sentence boundray
                        offset_span[1] = len(offset2token_map)
                    if offset_span[1] <= offset_span[0]:       # empty span
                        continue
                    span = [offset2token_map[offset_span[0]], offset2token_map[offset_span[1]-1]+1]

                    markups[annotype][wid].append(span)

        return markups


    def get_markups(self, annotype=None):
        if annotype == None:
            return self.markups
        elif annotype not in self.markups:
            return dict()
        else:
            return self.markups[annotype]


    def set_groundtruth(self, gt_markups_offset, gt_wids=None):
        self.groundtruth_offset = gt_markups_offset
        self.groundtruth = {}

        ## Combining groundtruth from multiple professionals
        markups = self.offset2markups(gt_markups_offset)
        self.groundtruth_markups = markups

        for annotype in markups:
            mask = [0] * self.ntokens

            for wid, spans in list(markups[annotype].items()):
                if gt_wids is not None and wid not in gt_wids:
                    continue

                for span in spans:
                    for i in range(span[0], span[1]):
                        mask[i] = 1

            self.groundtruth[annotype] = self._mask2spans(mask)


    def set_aggregation(self, agg_markups_offset, agg_ids=None):
        self.aggregation_offset = agg_markups_offset
        self.aggregation = {}

        ## Combining groundtruth from multiple professionals
        markups = self.offset2markups(agg_markups_offset)
        self.aggregation_markups = markups

        for annotype in markups:
            mask = [0] * self.ntokens

            for wid, spans in list(markups[annotype].items()):
                if agg_ids is not None and wid not in agg_ids:
                    continue

                for span in spans:
                    for i in range(span[0], span[1]):
                        mask[i] = 1

            self.aggregation[annotype] = self._mask2spans(mask)


    def get_groundtruth(self, annotype):
        if annotype == None or self.groundtruth == None:
            return self.groundtruth
        elif annotype not in self.groundtruth:
            return dict()
        else:
            return self.groundtruth[annotype]


    def get_aggregation(self, annotype):
        if annotype == None or self.aggregation == None:
            return self.aggregation
        elif annotype not in self.aggregation:
            return dict()
        else:
            return self.aggregation[annotype]

    def text(self):
        return self.spacydoc.text

    def tokenized_text(self):
        text = ""
        for token in self.spacydoc:
            text += " " + token.text
        return text.strip()

    def get_markups_text(self, annotype=None):
        if annotype and annotype not in self.markups:
            return dict()

        annotypes = [annotype] if annotype else list(self.markups.keys())

        markups_text = {}
        for annotype in annotypes:
            markups_anno = self.markups[annotype]
            markups_text[annotype] = {}

            for wid, spans in markups_anno.items():
                markups_text[annotype][wid] = []
                for span in spans:
                    text = self._get_text_by_span(span)
                    if len(text) > 0:
                        markups_text[annotype][wid].append((span, text))

        if annotype:
            return markups_text[annotype]
        else:
            return markups_text

    def _get_text_by_span(self, span):
        if span[1] <= span[0]:
            return ""
        return self.spacydoc[span[0]:span[1]].text

    def _mask2spans(self, mask):
        mask.append(0)  # append a non span

        spans = []
        if mask[0] == 1:
            sidx = 0

        for idx, v in enumerate(mask[1:], 1):
            if v==1 and mask[idx-1] == 0: # start of span
                sidx = idx
            elif v==0 and mask[idx-1] == 1 : # end of span
                eidx = idx
                spans.append( (sidx, eidx) )
        return spans


class Corpus:

    ANNOTYPES = ['Participants', 'Intervention', 'Outcome']

    # Cardiovascular | Cancer | Autistic
    TAG_GENRE_MAPPING = {
            # Autistic
            'Autistic_Disorder':'Autistic',
            'Child_Development_Disorders_Pervasive':'Autistic',
            'Psychiatric_Status_Rating_Scales':'Autistic',
            'Neuropsychological_Tests':'Autistic',
            'Behavior_Therapy':'Autistic',
            # Cardiovascular
            'Blood_Pressure':'Cardiovascular',
            'Hypertension':'Cardiovascular',
            'Heart_Rate':'Cardiovascular',
            'Body_Mass_Index':'Cardiovascular',
            'Myocardial_Infarction':'Cardiovascular',
            'Heart_Failure':'Cardiovascular',
            'Antihypertensive_Agents':'Cardiovascular',
            # Cancer
            'Antineoplastic_Combined_Chemotherapy_Protocols':'Cancer',
            'Breast_Neoplasms':'Cancer',
            'Antineoplastic_Agents':'Cancer',
            'Neoplasm_Staging':'Cancer',
            'Neoplasms':'Cancer',
            'Lung_Neoplasms':'Cancer',
            'Neoplasm_Recurrence_Local':'Cancer'
            }

    def __init__(self, doc_path, verbose=True):
        self.docs = dict()
        self.doc_path = doc_path
        self.verbose = verbose

    def __len__(self):
        return len(self.docs)

    def _process_anno_per_annotype(self, anno, max_num_worker, pruned_workers):
        anno_new = {}
        wids = [ wid for wid in list(anno.keys()) if wid not in pruned_workers ]
        wids.sort()
        random.shuffle(wids)

        if len(list(anno.keys())) != len(wids):
            print("filtered plus")

        if len(wids) > max_num_worker:
            wids = wids[:max_num_worker]

        for wid in wids:
            anno_new[wid] = anno[wid]

        return anno_new

    def _process_anno(self, anno, max_num_worker, pruned_workers={}):
        anno_new = {}

        max_num_worker =  1000 if max_num_worker is None else max_num_worker

        for key in list(anno.keys()):
            if key not in self.ANNOTYPES:
                anno_new[key] = anno[key]
            else:
                annotype = key

                anno_tmp = anno[annotype]
                pruned_workers_tmp = pruned_workers.get(annotype, [])
                anno_new_tmp = self._process_anno_per_annotype(\
                        anno_tmp, max_num_worker, pruned_workers_tmp)
                anno_new[annotype] = anno_new_tmp

        return anno_new

    def _get_doc_genres(self, docid):
        genre_fn = self.doc_path + '/mesh_tags/' + docid + '.txt'
        if not os.path.exists(genre_fn):
            print("mesh_tag {} file is not found in dataset, skip loading genre.".format(genre_fn))
            return set()

        genres = set()
        with open(genre_fn) as fin:
            for line in fin:
                tag = line.replace('*','').split('/')[0].strip().replace(' ', '_').replace(',','')
                if tag in self.TAG_GENRE_MAPPING:
                    genres.add(self.TAG_GENRE_MAPPING[tag])

        return genres

    def load_annotations(self, annos_fn, docids=None, max_num_worker=None, pruned_workers={}):
        with open(annos_fn) as fin:
            idx = 0
            for line in fin:
                idx += 1
                if idx % 500 == 0:
                    if self.verbose:
                        print('[INFO] {0} docs has been loaded'.format(idx))

                anno = json.loads(line.strip())
                docid = anno['docid']

                if docids != None and docid not in docids: # Skip doc not in the docids parameter
                    continue

                if max_num_worker or pruned_workers:
                    anno = self._process_anno(anno, max_num_worker, pruned_workers)

                doc_fn = self.doc_path + docid + '.txt'

                del anno['docid']

                if not os.path.exists(doc_fn):
                    raise Exception('{0} not found'.format(doc_fn))

                genres = self._get_doc_genres(docid)

                rawdoc = open(doc_fn).read()
                spacydoc = sp(rawdoc)
                self.docs[docid] = Doc(docid, anno, spacydoc, genres=genres)


    def load_groundtruth(self, gt_fn, gt_wids=None):
        """
        Load groundtruth for corpus, has to been called after load annotation
        """
        with open(gt_fn) as fin:
            for line in fin:
                anno = json.loads(line.strip())
                docid = anno['docid']
                del anno['docid']

                if docid not in self.docs:
                    if self.verbose:
                        print('[WARN] doc {0} is not loaded yet'.format(docid))
                    continue

                self.docs[docid].set_groundtruth(anno, gt_wids)


    def load_aggregation(self, agg_fn, agg_ids=None):
        """
        Load aggregated results for each doc
        """
        with open(agg_fn) as fin:
            for line in fin:
                aggs = json.loads(line.strip())
                docid = aggs['docid']
                del aggs['docid']

                if docid not in self.docs:
                    if self.verbose:
                        print('[WARN] doc {0} is not loaded yet'.format(docid))
                    continue

                self.docs[docid].set_aggregation(aggs, agg_ids)


    def get_doc_annos(self, docid, annotype=None, text=False):
        if docid not in self.docs:
            print('docid {0} is not found'.format(docid))
            return None

        if text:
            return self.docs[docid].get_markups_text(annotype)
        else:
            return self.docs[docid].get_markups(annotype)


    def get_doc_groundtruth(self, docid, annotype=None):
        if docid not in self.docs:
            print('docid {0} is not found'.format(docid))
            return None

        return self.docs[docid].get_groundtruth(annotype)


    def get_doc_aggregation(self, docid, annotype=None):
        if docid not in self.docs:
            print('docid {0} is not found'.format(docid))
            return None

        return self.docs[docid].get_aggregation(annotype)


    def get_doc_text(self, docid):
        if docid not in self.docs:
            print('docid {0} is not found'.format(docid))
            return None

        return self.docs[docid].text()


    def get_doc_tokenized_text(self, docid):
        if docid not in self.docs:
            print('docid {0} is not found'.format(docid))
            return None

        return self.docs[docid].tokenized_text()


    def get_doc_spacydoc(self, docid):
        if docid not in self.docs:
            print('docid {0} is not found'.format(docid))
            return None

        return self.docs[docid].spacydoc

    def get_doc_genres(self, docid):
        if docid not in self.docs:
            print('docid {0} is not found'.format(docid))
            return None
        return self.docs[docid].genres
