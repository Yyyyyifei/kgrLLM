import argparse
import ast
import collections
import logging
import torch
import os
import pickle

from dataloader import TestDataset
from huggingface_hub import login
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

from models import KGReasoning
from format_question import rel2text, load_info, hop_query_to_text, load_alignment_text

login("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")

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

def re_map_query(id2ent_original, ent2id_dest, id2rel_original, rel2id_dest, query):
    is_intersection = True
    for ele in query:
        if not isinstance(ele, tuple):
            is_intersection = False
            break

    if is_intersection:
        for q in query:
            q[0] = ent2id_dest[id2ent_original[q[0]]]
            q[1] = rel2id_dest[id2rel_original[q[1]]]
    else:
        query[0] = ent2id_dest[id2ent_original[query[0]]]
        for i, _ in enumerate(query[1]):
            query[1][i] = rel2id_dest[id2rel_original[query[1][i]]]
    
    return query

def obtain_ranking(model, test_dataloader, hard_answers, easy_answers, k):
    model.eval()

    first_k = {}
    all_rankings = {}
    answer_rankings = {}

    with torch.no_grad():
        for negative_sample, queries, queries_unflatten, query_structures in tqdm(test_dataloader, disable=False):
            batch_queries_dict = collections.defaultdict(list)
            batch_idxs_dict = collections.defaultdict(list)
            for i, query in enumerate(queries):
                batch_queries_dict[query_structures[i]].append(query)
                batch_idxs_dict[query_structures[i]].append(i)
            for query_structure in batch_queries_dict:
                batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
           
            negative_sample = negative_sample.cuda()

            _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
            queries_unflatten = [queries_unflatten[i] for i in idxs]
            query_structures = [query_structures[i] for i in idxs]
            
            argsort = torch.argsort(negative_logit, dim=1, descending=True)
            ranking = argsort.clone().to(torch.float)

            ranking = ranking.scatter_(1, 
                                        argsort, 
                                        torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                            1).cuda()
                                        ) # achieve the ranking of all entities

            first_k[queries_unflatten[0]] = argsort[0][:k]
            all_rankings[queries_unflatten[0]] = ranking

            for idx, (i, query, query_structure) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structures)):
                hard_answer = hard_answers[query]
                easy_answer = easy_answers[query]
                num_hard = len(hard_answer)
                num_easy = len(easy_answer)
                assert len(hard_answer.intersection(easy_answer)) == 0
                cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                cur_ranking, indices = torch.sort(cur_ranking)
                masks = indices >= num_easy

                answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()

                cur_ranking = cur_ranking - answer_list + 1 # filtered setting
                cur_ranking = cur_ranking[masks] # only take indices that belong to the hard answers

                answer_rankings[query] = cur_ranking

    return first_k, all_rankings, answer_rankings

# def load_data(tasks, id2ent_original, ent2id_dest, id2rel_original, rel2id_dest):
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

def load_model(checkpoint_path):
    model = KGReasoning(
        nentity=14505,
        nrelation=474,
        hidden_dim=800,
        gamma=12.0,
        geo='vec',
        use_cuda = True,
        test_batch_size=1,
        query_name_dict = query_name_dict
    )

    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    return model

def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries

def split_line(line):
    first_split = line.strip().find("\t")
    
    if first_split == -1:
        return None, None

    return line[:first_split], line[first_split+1:]

def query_to_text(id2ent, id2rel, ent2text, query, query_type):
    target = []

    if isinstance(query_type, list):
        for i, ele in enumerate(query_type):
            target.append(query_to_text(id2ent, id2rel, ent2text, query[i], ele))
    else:
        assert isinstance(query_type, str), "invalid type element"
        assert isinstance(query, int), "invalid query id"

        if query_type == "e":
            return ent2text[id2ent[query]].strip()
        else:
            assert query_type == "r", "invalid element type"
            return rel2text(id2rel[query])
        
    return target

def llama3_model_set_up():
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return tokenizer, model

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Parser for running different kinds of logical query")

    parser.add_argument('-type', "--type", default="2p", help='type of query we want to run on')
    parser.add_argument('-k', "--retrieve_k", default=10, help="how many entities we want to retrieve")

    return parser.parse_args(args)

def tuple_to_list(query):
    target = []
    if isinstance(query, tuple):
        for ele in query:
            target.append(tuple_to_list(ele))
    else:
        return query
    
    return target

def main():
    args = parse_args()

    model = load_model("/h/224/yfsun/KGReasoning/logs/FB15k-237-q2b/1p.2p.3p.2i.3i/vec/g-24.0/2024.06.14-19:02:12/final_model")

    model.cuda()
    test_queries, test_hard_answers, test_easy_answers = load_data([args.type])
    alignment_data = load_alignment_text("/h/224/yfsun/KGReasoning/alignment_clean.json")

    test_dataloader = DataLoader(
        TestDataset(
            flatten_query(test_queries), 
            14505, 
            474, 
        ), 
        batch_size=1,
        num_workers=8, 
        collate_fn=TestDataset.collate_fn
    )

    first_k, all_rankings, answer_rankings = obtain_ranking(model, test_dataloader, test_hard_answers, test_easy_answers, args.retrieve_k)

    id2ent, id2rel, ent2text = load_info()

    torch.cuda.empty_cache()  

    tokenizer, llama3_model = llama3_model_set_up()

    llama3_model.to("cuda")

    raw_MRR = 0
    raw_1 = 0
    raw_3 = 0

    MRR = 0
    Hit_At_1 = 0
    Hit_At_3 = 0

    count = 0

    for query in tqdm(all_rankings):
        text_question = hop_query_to_text(alignment_data, id2ent, id2rel, ent2text, query)

        if not text_question:
            continue

        if count > 200:
            break

        answer_text = [ent2text[id2ent[answer.item()]].strip() for answer in first_k[query]]

        all_answers = test_hard_answers[query].copy()
        all_answers.update(test_easy_answers[query])

        hard_answers_text = [ent2text[id2ent[answer]].strip() for answer in test_hard_answers[query]]
        all_answers_text = [ent2text[id2ent[answer]].strip() for answer in all_answers]

        logging.info(f"Pre-reranked entities {answer_text}")
        
        messages = [
            {
                "role": "system", 
                "content": "You are a evaluator tasked with re-ranking entities for [Answer] masked in the question. [I_n] masks represent intermediate nodes. You only response with the ranking numbers without explanation"
            },
            # ranking-based
            {
                "role": "user", 
                "content": f"The question is {text_question}. The answer pool is {answer_text} (no ordering information contained). \
                    Please re-rank all of the entities provided and give ranking of each entity in one python list. \
                    Entities can have the same rannking and treat entities that has the same ranking as one entity"
            },

            # text-based
            # {
            #     "role": "user", 
            #     "content": f"The question is {text_question}. The answer pool is {answer_text} (no ordering information contained). \
            #         Please re-rank all of the entities provided and give the results . \
            #         Place entities you think that will rank the same in one nested list."
            # This line is only for intermediate nodes prediction.
            # And predict the intermediate nodes
            # },
        ]

        logging.info(messages)

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(llama3_model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = llama3_model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.2,
            top_p=0.1,
        )
        response = outputs[0][input_ids.shape[-1]:]

        llama3_answer = tokenizer.decode(response, skip_special_tokens=True)

        logging.info(f"llama3 return answer {llama3_answer}")

        re_ranked_entities = ast.literal_eval(llama3_answer)

        logging.info(f"All answers include {all_answers_text}")
        logging.info(f"Pre-ranked answer {answer_rankings[query]}")

        logging.info("")
        
        raw_mrr = 0
        raw_hit_1 = 0
        raw_hit_3 = 0

        mrr = 0
        hit_at_1 = 0
        hit_at_3 = 0

        if len(hard_answers_text) == 0:
            continue

        for i, ent in enumerate(hard_answers_text):
            new_answer_text = all_answers_text.copy()
            new_answer_text.remove(ent)

            re_ranking = [int(rnk) for rnk in re_ranked_entities]

            raw_ranking = answer_rankings[query][i]

            #[TODO: Apply filtered settings]
            if len(re_ranking) == args.retrieve_k and ent in answer_text:
                ranking = re_ranking[answer_text.index(ent)]
            else:
                ranking = answer_rankings[query][i]

            mrr += 1 / ranking
            hit_at_1 += 1 if ranking == 1 else 0
            hit_at_3 += 1 if ranking <= 3 else 0

            raw_mrr += 1 / raw_ranking
            raw_hit_1 += 1 if raw_ranking == 1 else 0
            raw_hit_3 += 1 if raw_ranking <= 3 else 0

        answer_length = len(hard_answers_text)

        MRR +=mrr / answer_length
        Hit_At_1 += hit_at_1 / answer_length
        Hit_At_3 += hit_at_3 / answer_length

        raw_MRR += raw_mrr / answer_length
        raw_1 += raw_hit_1 / answer_length
        raw_3 += raw_hit_3 / answer_length

        count += 1
        
    print(MRR.item() / count)
    print(Hit_At_1 / count)
    print(Hit_At_3 / count)

    print("")

    print(raw_MRR.item() / count)
    print(raw_1 / count)
    print(raw_3 / count)

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger("transformers").setLevel(logging.ERROR)

    # Suppress warnings
    import warnings
    warnings.filterwarnings("ignore")

    main()
