import pickle
import json

query_name_dict = {('e',('r',)): '1p', 
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys())

head_symbol = "[H]"
tail_symbol = "[T]"

def load_data(tasks):
    '''
    Load queries and remove queries not in tasks
    '''
    test_queries = pickle.load(open("/h/224/yfsun/GQE_Implementation/FB15k-237-q2b/test-queries.pkl", 'rb'))
    
    test_hard_answers = pickle.load(open("/h/224/yfsun/GQE_Implementation/FB15k-237-q2b/test-easy-answers.pkl", 'rb'))
    test_easy_answers = pickle.load(open("/h/224/yfsun/GQE_Implementation/FB15k-237-q2b/test-hard-answers.pkl", 'rb'))
    
    for name in all_tasks:
        if name not in tasks:
            query_structure = name_query_dict[name]
            if query_structure in test_queries:
                del test_queries[query_structure]

    return test_queries, test_hard_answers, test_easy_answers

def split_line(line):
    first_split = line.strip().find("\t")
    
    if first_split == -1:
        return None, None

    return line[:first_split], line[first_split+1:]

def load_info():
    id2ent = pickle.load(open("/h/224/yfsun/GQE_Implementation/FB15k-237-q2b/id2ent.pkl", "rb"))
    id2rel = pickle.load(open("/h/224/yfsun/GQE_Implementation/FB15k-237-q2b/id2rel.pkl", "rb"))

    ent2text = {}

    with open("/h/224/yfsun/GQE_Implementation/data/FB15k-237/entity2text.txt", "r") as f:
        for line in f.readlines():
            key, value = split_line(line)
            ent2text[key] = value

    return id2ent, id2rel, ent2text

def rel2text(rel):
    rel = rel.strip("/")

    segments = ["/", "_"]

    for seg in segments:
        rel = rel.replace(seg, " ")
    
    return rel

def rel_to_alignment_text(data, relid, id2rel):
    raw_relation = id2rel[relid].split("/")

    reverse_symbol = raw_relation[0]
    
    raw_text = "/" + "/".join(raw_relation[1:])

    text_relation = data[raw_text]
    
    text_relation = text_relation.replace(head_symbol, "{head}")
    text_relation = text_relation.replace(tail_symbol, "{tail}")

    return reverse_symbol, text_relation.strip()

def load_alignment_text(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    return data

def hop_query_to_text(alignment_data, id2ent, id2rel, ent2text, query):
    assert not isinstance(query[0], tuple)

    head_entity = ent2text[id2ent[query[0]]].strip()

    relations = []
    
    for relation in query[1]:
        reverse_symbol, text_relation = rel_to_alignment_text(alignment_data, relation, id2rel)
        relations.append(text_relation)

        if reverse_symbol == "-":
            return ""
    
    head_used = False

    final_question = ""
    num_intermediate = 0

    for rel in relations:
        if not head_used:
            num_intermediate += 1
            formatted_rel = rel.format(head=head_entity, tail=f"[I_{num_intermediate}]")
            final_question += formatted_rel

            head_used = True
        else:
            formatted_rel = rel.format(head=f"[I_{num_intermediate}]", tail=f"[I_{num_intermediate+1}]")
            num_intermediate += 1
            final_question += formatted_rel

    final_question = final_question.replace(f"[I_{num_intermediate}]", "[Answer]")

    return final_question

def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries

def query_to_text(id2ent, id2rel, ent2text, query, query_type):
    target = []

    if isinstance(query_type, list):
        for i, ele in enumerate(query_type):
            target.append(query_to_text(id2ent, id2rel, ent2text, query[i], ele))
    else:
        assert isinstance(query_type, str), "invalid type element"
        assert isinstance(query, int), "invalid query id"

        if query_type == "e":
            return ent2text[id2ent[query]]
        else:
            assert query_type == "r", "invalid element type"
            return rel2text(id2rel[query])
        
    return target

if __name__ == "__main__":

    use_alignment = True

    test_queries, test_hard_answers, test_easy_answers = load_data("2p 3p")

    id2ent, id2rel, ent2text = load_info()
    
    alignment_data = load_alignment_text("/h/224/yfsun/KGReasoning/alignment_clean.json")

    for i, query in enumerate(test_queries[('e', ('r', 'r'))]):
        print(hop_query_to_text(alignment_data, id2ent, id2rel, ent2text, query))

        if i > 10:
            break
    
    print()

    for i, query in enumerate(test_queries[("e", ("r", "r", "r"))]):
        print(hop_query_to_text(alignment_data, id2ent, id2rel, ent2text, query))

        if i > 10:
            break