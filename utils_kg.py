import numpy as np
import torch

import rdflib
from rdflib import Graph


def create_y_emb(y_train, y_test, class_label_mapping, g_emb):
    '''
    Create embedding for the target vector based on KG embedding.

    Args:
        y_train: Train target vector
        y_test: Test target vector
        class_label_mapping: Mapping dictionary between labels,
            entities and indices
        g_emb: KG embedding

    Returns:
        y_train_emb: Train target vector embedding
        y_test_emb: Test target vector embedding
    '''
    if not isinstance(g_emb, torch.Tensor):
        g_emb = torch.tensor(g_emb)
    y_train_emb = torch.stack([
        g_emb[class_label_mapping['ent'][lab_e]]
        for i in y_train
        for lab_d in class_label_mapping['data_target'].keys()
        for lab_e in class_label_mapping['ent'].keys()
        if (i == class_label_mapping['data_target'][lab_d]) & (lab_d == lab_e)
    ])
    y_test_emb = torch.stack([
        g_emb[class_label_mapping['ent'][lab_e]]
        for i in y_test
        for lab_d in class_label_mapping['data_target'].keys()
        for lab_e in class_label_mapping['ent'].keys()
        if (i == class_label_mapping['data_target'][lab_d]) & (lab_d == lab_e)
    ])
    return y_train_emb, y_test_emb


def get_kg_embedding(g_file):
    '''
    Create knowledge graph embedding based on config.

    Args:
        g_file: KG file name

    Returns:
        g_emb: KG embedding
        ent_to_index: Dictionary for mapping entities to embedding
    '''
    # Load KG
    g = Graph()
    g_in = g.parse(g_file)

    g_emb, ent_to_index = subclass_embedding(g_in)

    return g_emb, ent_to_index


def subclass_embedding(g_in):
    '''
    Create KG embedding based on rdfs:subClassOf relationships.

    Args:
        g_in: KG

    Returns:
        g_emb: Manual subclass embedding of knowledge graph
        ent_to_index: Dictionary for mapping entities to indeces of
            sub_class_matrix
    '''
    # Query transitive subClassOf relationship
    query = list(g_in.query("""
        SELECT ?s ?o
        WHERE{
            ?s rdfs:subClassOf+ ?o
        }
    """))

    # Exclude dummy entity
    entities = [
        i
        for i in sorted(list(g_in.all_nodes()))
        if i != rdflib.term.URIRef('http://bearingfault.org/o_0')
    ]
    n_ent = len(entities)
    ent_to_index = {node: i for i, node in enumerate(entities)}

    # Create embedding matrix
    g_emb = np.zeros((n_ent, n_ent))
    for s, o in query:
        i = ent_to_index[s]
        j = ent_to_index[o]
        g_emb[i, j] = 1

    # All classes are subclasses of themselves
    g_emb = g_emb + np.diag(np.ones(n_ent))

    return g_emb, ent_to_index


def match_kg_data_labels(class_label_mapping, ent_to_index):
    '''
    Match keys between KG embedding and data labels.

    Args:
        class_label_mappping: Data label mapping
        ent_to_index: KG embedding mapping

    Returns
        class_label_mapping: Original class_label_mapping including
            KG embedding to data labels mapping
    '''
    pos_g_emb = [(i, ent_to_index[k])
                 for i in class_label_mapping['data_target'].keys()
                 for k in ent_to_index.keys()
                 if i in k]
    class_label_mapping['ent'] = {
        f_ins[0]: f_ins[1] for f_ins in pos_g_emb
    }
    return class_label_mapping
