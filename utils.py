import os, re, json, sys, codecs
from pprint import pprint


def finalize(chunks, clusters, ner, srls, zeros, genre):
    doc_data = {}
    sentences = [[word['wd'] for chunk in sentence for word in chunk.words ]for sentence in chunks]
    speakers = [['Speaker#1' for chunk in sentence for word in chunk.words] for sentence in chunks]

    doc_data['sentences'] = sentences
    doc_data['clusters'] = clusters
    doc_data['ner'] = ner
    doc_data['doc_key'] = genre
    doc_data['constituents'] = []
    doc_data['speakers'] = speakers
    doc_data['srl'] = srls
    doc_data['zero_clusters'] = zeros

    return doc_data

