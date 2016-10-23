'''
Created on Oct 23, 2016

@author: Melvin Laux
'''

def check_document_syntax(document):
    for i in xrange(len(document)-1):
        
        # check for invalid labels
        if document[i] not in [0,1,2]:
            print 'found invalid label', document
            return False
        
        # check for forbidden transitions
        if ((document[i]==1) and (document[i+1]==0)):
            print 'found forbidden transition', document
            return False 
        
    return True