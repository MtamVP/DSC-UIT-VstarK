import os
import re
import json
import zipfile
import pickle
from collections import defaultdict
import numpy as np
import torch
from optimum.onnxruntime import ORTModelForSequenceClassification
from onnxruntime import SessionOptions, GraphOptimizationLevel
from transformers import AutoTokenizer, pipeline
import warnings
warnings.filterwarnings("ignore")


# Configuration
DENSE_TOP_K = 300           # dense search top-k chunks
BM25_CHUNK_TOP_K = 300      # BM25 chunk search top-k
BM25_NAME_TOP_K = 100       # BM25 name search top-k
BM25_PASSAGE_TOP_K = 200    # BM25 full-passage search top-k
TASK1_K_CANDIDATES = 50     # Reduced from 200 to speed up by 4x
TASK2_K_CANDIDATES = 20     # Reduced from 60 to speed up

RRF_K = 40                  # RRF constant — smaller = boost top results more
W_BM25_CHUNK = 1.5          # RRF weight for BM25 chunk
W_BM25_NAME = 1.2           # RRF weight for BM25 name
W_BM25_PASSAGE = 1.0        # RRF weight for BM25 full-passage
W_DENSE = 2.0               # RRF weight for dense
NON_MATCH_RANK = 500        # default rank for non-matching docs
CITATION_BONUS = 0.25       # citation match bonus
TITLE_MATCH_BONUS = 0.15    # title keyword overlap bonus
MULTI_HIT_BONUS_PER = 0.02  # bonus per BM25 chunk hit (capped)
MULTI_HIT_BONUS_MAX = 0.10  # max multi-hit bonus

RERANKER_CTX_LEN = 3000     
CE_WEIGHT = 1.0             # cross-encoder score weight
RRF_WEIGHT = 5.0           # RRF score weight in final combination
TASK1_RRF_SAFETY_K = 5      # keep top-K by RRF as safety net
TASK1_CE_SAFETY_K = 5       # keep top-K by CE as safety net

ARTICLE_MAX_LEN = 2500      # max chars per article chunk
FIXED_CHUNK_STEP = 800      # fixed-length chunk step
FIXED_CHUNK_WINDOW = 1200   # fixed-length chunk window
EMBED_TEXT_LEN = 800         # text length for embedding

TASK2_MAX_ANSWER_LEN = 5000  # max answer length for task2

EMBEDDER_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
QA_MODEL = "nguyenvulebinh/vi-mrc-large"

CACHE_CORPUS = "corpus_cache_v3.pkl"
CACHE_INDEX = "dense_index_v3.faiss"


# Text Processing
def tokenize_vi(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())


def clean_title(name: str) -> str:
    return re.sub(r'[\-_]+', ' ', name).strip()


def extract_citations(text: str) -> set[str]:
    """Extract legal document citation identifiers from text."""
    patterns = [
        r'(?:Nghị\s*định|NĐ)\s*(?:số)?\s*(\d+(?:/\d+)*(?:/[A-ZĐa-zđ\-]+)?)',
        r'(?:Thông\s*tư|TT)\s*(?:số)?\s*(\d+(?:/\d+)*(?:/[A-ZĐa-zđ\-]+)?)',
        r'(?:Quyết\s*định|QĐ)\s*(?:số)?\s*(\d+(?:/\d+)*(?:/[A-ZĐa-zđ\-]+)?)',
        r'(?:Luật|Bộ\s*luật)\s*(?:số)?\s*(\d+(?:/\d+)*(?:/[A-ZĐa-zđ\-]+)?)',
        r'(?:Nghị\s*quyết|NQ)\s*(?:số)?\s*(\d+(?:/\d+)*(?:/[A-ZĐa-zđ\-]+)?)',
        r'(?:Chỉ\s*thị|CT)\s*(?:số)?\s*(\d+(?:/\d+)*(?:/[A-ZĐa-zđ\-]+)?)',
        r'(?:Công\s*văn|CV)\s*(?:số)?\s*(\d+(?:/\d+)*(?:/[A-ZĐa-zđ\-]+)?)',
        r'(?:Thông\s*báo|TB)\s*(?:số)?\s*(\d+(?:/\d+)*(?:/[A-ZĐa-zđ\-]+)?)',
    ]
    citations = set()
    for pattern in patterns:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            clean = re.sub(r'[\s\-]', '', match.lower())
            if len(clean) >= 2:
                citations.add(clean)
    return citations


def extract_dieu_references(text: str) -> set[str]:
    """Extract 'Điều N' references from question text."""
    return set(re.findall(r'Điều\s+(\d+)', text))


# Chunking
def split_articles(doc_id: str, doc_name: str, doc_text: str,
                   max_len: int = ARTICLE_MAX_LEN) -> list[dict]:
    """Split document into chunks based on article boundaries."""
    title = clean_title(doc_name)
    matches = list(re.finditer(r'(?i)(?:^|\n)\s*(Điều\s+\d+[\.:\s])', doc_text))
    chunks = []

    if matches:
        if matches[0].start() > 50:
            preamble = doc_text[:matches[0].start()].strip()
            if preamble:
                chunks.append({
                    "doc_id": doc_id,
                    "text": f"{title}: {preamble[:max_len]}",
                    "raw_text": preamble
                })

        for i in range(len(matches)):
            start = matches[i].start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(doc_text)
            art_text = doc_text[start:end].strip()
            art_number = matches[i].group(1).strip(' .:').strip()
            if art_text:
                chunks.append({
                    "doc_id": doc_id,
                    "text": f"[{title} > {art_number}] {art_text[:max_len]}",
                    "raw_text": art_text
                })
                if len(art_text) > max_len:
                    for j in range(max_len - 600, len(art_text), max_len - 600):
                        sub = art_text[j:j + max_len].strip()
                        if len(sub) > 100:
                            chunks.append({
                                "doc_id": doc_id,
                                "text": f"[{title} > {art_number} (tiếp)] {sub}",
                                "raw_text": sub
                            })
    else:
        for i in range(0, len(doc_text), FIXED_CHUNK_STEP):
            chunk = doc_text[i:i + FIXED_CHUNK_WINDOW].strip()
            if len(chunk) > 20:
                chunks.append({
                    "doc_id": doc_id,
                    "text": f"{title}: {chunk}",
                    "raw_text": chunk
                })

    if not chunks:
        chunks.append({
            "doc_id": doc_id,
            "text": f"{title}: {doc_text[:max_len]}",
            "raw_text": doc_text
        })
    return chunks


# Corpus Loading
def load_corpus(zip_path: str, fallback_dir: str) -> dict:
    corpus = {}
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for name in zf.namelist():
                if name.endswith('.json') and not name.startswith('__'):
                    did = os.path.splitext(os.path.basename(name))[0].replace("context_", "")
                    content = json.loads(zf.read(name).decode('utf-8'))
                    item = content[0] if isinstance(content, list) and content else content if isinstance(content, dict) else {}
                    if item:
                        corpus[did] = {
                            "id": did,
                            "name": item.get("name") or item.get("link", "").split("/")[-1].replace(".aspx", "").replace(".html", ""),
                            "passage": item.get("passage", "")
                        }
    elif os.path.exists(fallback_dir):
        for fname in os.listdir(fallback_dir):
            if fname.endswith('.json'):
                fp = os.path.join(fallback_dir, fname)
                did = os.path.splitext(fname)[0].replace("context_", "")
                with open(fp, encoding="utf-8") as f:
                    content = json.load(f)
                    item = content[0] if isinstance(content, list) and content else content if isinstance(content, dict) else {}
                    if item:
                        corpus[did] = {
                            "id": did,
                            "name": item.get("name") or item.get("link", "").split("/")[-1].replace(".aspx", "").replace(".html", ""),
                            "passage": item.get("passage", "")
                        }
    return corpus


def load_queries(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return {str(item["id"]): item["question"] for item in data}
    return {str(k): v["question"] for k, v in data.items()}


# BM25 Engine
class UltraFastBM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5,
                 b: float = 0.75, min_idf: float = 0.5):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_len = np.array([len(doc) for doc in corpus_tokens], dtype=np.int32)
        self.avgdl = float(np.mean(self.doc_len)) if self.corpus_size > 0 else 1.0

        postings = defaultdict(lambda: ([], []))
        for doc_id, doc in enumerate(corpus_tokens):
            tf_dict = defaultdict(int)
            for token in doc:
                tf_dict[token] += 1
            for token, tf in tf_dict.items():
                docs, tfs = postings[token]
                docs.append(doc_id)
                tfs.append(tf)

        self.inverted_index = {}
        for token, (docs, tfs) in postings.items():
            df = len(docs)
            idf = np.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            if idf > min_idf and df < 0.35 * self.corpus_size:
                docs_arr = np.array(docs, dtype=np.int32)
                tfs_arr = np.array(tfs, dtype=np.float32)
                doc_lens = self.doc_len[docs_arr]
                denom = tfs_arr + self.k1 * (1.0 - self.b + self.b * (doc_lens / self.avgdl))
                precomputed = (idf * (tfs_arr * (self.k1 + 1.0)) / denom).astype(np.float32)
                self.inverted_index[token] = (docs_arr, precomputed)

    def get_top_k(self, query_tokens: list[str], k: int = 50) -> list[tuple]:
        doc_scores = np.zeros(self.corpus_size, dtype=np.float32)
        has_match = False
        
        for token in query_tokens:
            if token in self.inverted_index:
                docs, scores = self.inverted_index[token]
                doc_scores[docs] += scores
                has_match = True
                
        if not has_match:
            return []
            
        top_k = min(k, self.corpus_size)
        top_indices = np.argpartition(doc_scores, -top_k)[-top_k:]
        
        top_indices = top_indices[np.argsort(-doc_scores[top_indices])]
        
        result = [(int(idx), float(doc_scores[idx])) for idx in top_indices if doc_scores[idx] > 0]
        return result


# Hybrid Retrieval with RRF Fusion
def retrieve(queries_dict, corpus, chunks, doc_ids, doc_names, doc_citations,
             bm25_chunk, bm25_name, bm25_passage, embedder, dense_index, reranker,
             k_candidates=100, use_safety_net=False, task_name="Task1"):
    """Retrieve relevant documents for each query using hybrid BM25 + dense + reranker."""
    qids = list(queries_dict.keys())
    qtexts = [queries_dict[qid] for qid in qids]

    print("Build doc_to_chunks mapping for CE")
    doc_to_chunks = defaultdict(list)
    for c in chunks:
        doc_to_chunks[c["doc_id"]].append(c["text"])

    doc_name_tokens = {did: set(tokenize_vi(name)) for did, name in doc_names.items()}

    citation_inv_index = defaultdict(set)
    for did, cites in doc_citations.items():
        for c in cites:
            citation_inv_index[c].add(did)

    title_token_inv = defaultdict(set)
    for did, toks in doc_name_tokens.items():
        for t in toks:
            title_token_inv[t].add(did)

    dense_top_chunk_idxs = [[] for _ in qids]

    all_pairs = []
    pair_metadata = []
    doc_chunk_map = defaultdict(dict)

    for q_i, qid in enumerate(qids):
        qtext = qtexts[q_i]
        q_toks = tokenize_vi(qtext)
        q_cites = extract_citations(qtext)
        q_toks_set = set(q_toks)

        top_chunks = bm25_chunk.get_top_k(q_toks, k=BM25_CHUNK_TOP_K)
        bm25_chunk_ranks = {}
        bm25_chunk_hits = defaultdict(int)
        top_chunk_per_doc = {}
        for rank, (c_idx, score) in enumerate(top_chunks):
            did = chunks[c_idx]["doc_id"]
            bm25_chunk_hits[did] += 1
            if did not in bm25_chunk_ranks:
                bm25_chunk_ranks[did] = rank + 1
                top_chunk_per_doc[did] = chunks[c_idx]["text"]
            if did not in doc_chunk_map[qid]:
                doc_chunk_map[qid][did] = []
            if len(doc_chunk_map[qid][did]) < 2:
                doc_chunk_map[qid][did].append(chunks[c_idx]["text"])

        top_names = bm25_name.get_top_k(q_toks, k=BM25_NAME_TOP_K)
        bm25_name_ranks = {
            doc_ids[n_idx]: rank + 1
            for rank, (n_idx, score) in enumerate(top_names)
        }

        bm25_passage_ranks = {}
        if bm25_passage is not None:
            top_passages = bm25_passage.get_top_k(q_toks, k=BM25_PASSAGE_TOP_K)
            for rank, (p_idx, score) in enumerate(top_passages):
                did = doc_ids[p_idx]
                if did not in bm25_passage_ranks:
                    bm25_passage_ranks[did] = rank + 1

        dense_chunk_ranks = {}
        for rank, c_idx in enumerate(dense_top_chunk_idxs[q_i]):
            did = chunks[c_idx]["doc_id"]
            if did not in dense_chunk_ranks:
                dense_chunk_ranks[did] = rank + 1
                if did not in top_chunk_per_doc:
                    top_chunk_per_doc[did] = chunks[c_idx]["text"]
                if did not in doc_chunk_map[qid]:
                    doc_chunk_map[qid][did] = []
                if len(doc_chunk_map[qid][did]) < 2:
                    doc_chunk_map[qid][did].append(chunks[c_idx]["text"])
        cite_matches = set()
        if q_cites:
            for cite_key in q_cites:
                cite_matches.update(citation_inv_index.get(cite_key, set()))

        title_matches = set()
        if len(q_toks_set) >= 3:
            title_hit_counts = defaultdict(int)
            for tok in q_toks_set:
                for did in title_token_inv.get(tok, set()):
                    title_hit_counts[did] += 1
            min_overlap = min(3, max(1, int(len(q_toks_set) * 0.35)))
            for did, cnt in title_hit_counts.items():
                if cnt >= min_overlap:
                    title_matches.add(did)

        candidates = (
            set(bm25_chunk_ranks) | set(bm25_name_ranks) |
            set(bm25_passage_ranks) | set(dense_chunk_ranks) |
            cite_matches | title_matches
        )

        fused = {}
        for did in candidates:
            score = (
                W_BM25_CHUNK / (RRF_K + bm25_chunk_ranks.get(did, NON_MATCH_RANK)) +
                W_BM25_NAME / (RRF_K + bm25_name_ranks.get(did, NON_MATCH_RANK)) +
                W_BM25_PASSAGE / (RRF_K + bm25_passage_ranks.get(did, NON_MATCH_RANK)) +
                W_DENSE / (RRF_K + dense_chunk_ranks.get(did, NON_MATCH_RANK))
            )
            if did in cite_matches:
                score += CITATION_BONUS
            if did in title_matches:
                score += TITLE_MATCH_BONUS
            multi_hit = min(bm25_chunk_hits.get(did, 0) * MULTI_HIT_BONUS_PER, MULTI_HIT_BONUS_MAX)
            score += multi_hit

            fused[did] = score

        top_candidates = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k_candidates]
        for did, rrf_score in top_candidates:
            chunk_list = doc_chunk_map[qid].get(did, [])
            if not chunk_list:
                if did in top_chunk_per_doc:
                    chunk_list = [top_chunk_per_doc[did]]
                else:
                    chunk_list = [corpus[did]["passage"][:2000]]
            
            chunk_list = list(dict.fromkeys(chunk_list))
            
            combined_ctx = ""
            for c in chunk_list:
                if len(combined_ctx) + len(c) < RERANKER_CTX_LEN:
                    combined_ctx += c + "\n"
            if not combined_ctx: combined_ctx = chunk_list[0]
            
            pair_metadata.append((qid, did, rrf_score, combined_ctx))
            all_pairs.append((qtext, combined_ctx[:RERANKER_CTX_LEN]))

    print(f"Cross-encode {len(all_pairs)} pairs")
    ce_model, ce_tokenizer = reranker
    batch_size = 64
    checkpoint_file = f"{task_name}_ce_checkpoint.json"
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            all_ce_scores = json.load(f)
        start_idx = len(all_ce_scores)
        print(f"Resuming from checkpoint: {start_idx}/{len(all_pairs)} pairs completed.")
    else:
        all_ce_scores = []
        start_idx = 0
        
    for i in range(start_idx, len(all_pairs), batch_size):
        batch_pairs = all_pairs[i:i+batch_size]
        inputs = ce_tokenizer(batch_pairs, padding=True, truncation=True, max_length=512, return_tensors="np")
        outputs = ce_model(**inputs)
        scores = outputs.logits.squeeze(-1).tolist()
        if isinstance(scores, float):
            scores = [scores]
        all_ce_scores.extend(scores)
        
        # Save checkpoint every 50 batches (3200 pairs)
        if (i // batch_size) % 50 == 0:
            with open(checkpoint_file, "w") as f:
                json.dump(all_ce_scores, f)
                
    with open(checkpoint_file, "w") as f:
        json.dump(all_ce_scores, f)

    doc_scores = defaultdict(lambda: defaultdict(lambda: {"ce": -999, "rrf": 0, "chunk": ""}))
    for idx, (qid, did, rrf_score, chunk_text) in enumerate(pair_metadata):
        ce = float(all_ce_scores[idx])
        entry = doc_scores[qid][did]
        if ce > entry["ce"]:
            entry["ce"] = ce
            entry["chunk"] = chunk_text
        entry["rrf"] = rrf_score

    results = {}
    for qid in doc_scores:
        scored_list = []
        for did, entry in doc_scores[qid].items():
            combined = entry["ce"] * CE_WEIGHT + RRF_WEIGHT * entry["rrf"]
            scored_list.append((did, combined, entry["rrf"], entry["ce"], entry["chunk"]))

        if use_safety_net:
            by_rrf = sorted(scored_list, key=lambda x: x[2], reverse=True)
            by_ce = sorted(scored_list, key=lambda x: x[3], reverse=True)
            
            final_docs = []
            final_dids = set()
            
            for i in range(max(len(by_ce), len(by_rrf))):
                if i < len(by_ce):
                    d = by_ce[i]
                    if d[0] not in final_dids:
                        final_dids.add(d[0])
                        final_docs.append((d[0], d[1], d[4]))
                
                if i < len(by_rrf):
                    d = by_rrf[i]
                    if d[0] not in final_dids:
                        final_dids.add(d[0])
                        final_docs.append((d[0], d[1], d[4]))
                        
            results[qid] = final_docs
        else:
            scored_list.sort(key=lambda x: x[1], reverse=True)
            results[qid] = [(d[0], d[1], d[4]) for d in scored_list]

    return results


# Task 2: Smart Answer Extraction (Maximize METEOR)
def extract_answer_for_qa(question: str, doc_id: str, corpus: dict,
                          max_chars: int = TASK2_MAX_ANSWER_LEN) -> str:
    """
    Extract the most relevant answer from a document's passage.

    Strategy for maximizing METEOR:
    - METEOR weighs recall 9x more than precision → return MORE text
    - Reference answers are typically extractive from passages
    - So returning the right passage section = high METEOR
    """
    passage = corpus.get(doc_id, {}).get("passage", "")
    if not passage:
        return ""

    if len(passage) <= max_chars:
        return passage.strip()

    q_tokens = set(tokenize_vi(question))
    q_dieu_refs = extract_dieu_references(question)
    article_matches = list(re.finditer(r'(?i)(?:^|\n)\s*(Điều\s+(\d+)[\.:\s])', passage))

    if article_matches:
        articles = []
        for i in range(len(article_matches)):
            start = article_matches[i].start()
            end = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(passage)
            art_text = passage[start:end].strip()
            art_number = article_matches[i].group(2)

            art_tokens = set(tokenize_vi(art_text))
            token_overlap = len(q_tokens & art_tokens)

            dieu_bonus = 15 if art_number in q_dieu_refs else 0

            score = token_overlap + dieu_bonus
            articles.append((score, start, end, art_text))

        articles.sort(key=lambda x: x[0], reverse=True)

        result_parts = []
        total_len = 0

        if article_matches[0].start() > 50:
            preamble = passage[:article_matches[0].start()].strip()
            preamble_tokens = set(tokenize_vi(preamble))
            preamble_overlap = len(q_tokens & preamble_tokens)
            if preamble_overlap >= 2 and len(preamble) < max_chars * 0.3:
                result_parts.append((0, preamble))
                total_len += len(preamble) + 2

        for score, start, end, art_text in articles:
            if total_len + len(art_text) > max_chars:
                remaining = max_chars - total_len
                if remaining > 200:
                    result_parts.append((start, art_text[:remaining]))
                    total_len += remaining
                break
            result_parts.append((start, art_text))
            total_len += len(art_text) + 2

        if total_len < max_chars * 0.5:
            added_starts = {p[0] for p in result_parts}
            for score, start, end, art_text in sorted(articles, key=lambda x: x[1]):
                if start in added_starts:
                    continue
                if total_len + len(art_text) > max_chars:
                    break
                result_parts.append((start, art_text))
                total_len += len(art_text) + 2

        result_parts.sort(key=lambda x: x[0])
        return "\n".join([p[1] for p in result_parts]).strip()

    return passage[:max_chars].strip()


def extract_multi_doc_answer(question: str, scored_docs: list, corpus: dict,
                              qa_pipeline=None, max_chars: int = TASK2_MAX_ANSWER_LEN) -> str:
    """
    Extract answer from top-ranked documents.
    Uses QA pipeline to find anchor exact span, then expands.
    """
    if not scored_docs:
        return ""

    top_doc_id = scored_docs[0][0]
    chunk_text = scored_docs[0][2]
    
    # Run QA model to find anchor
    qa_span = ""
    if qa_pipeline is not None:
        try:
            res = qa_pipeline(question=question, context=chunk_text[:1000])
            qa_span = res.get('answer', '')
        except Exception:
            pass

    answer = extract_answer_for_qa(question, top_doc_id, corpus, max_chars)

    if qa_span and qa_span not in answer:
        answer = qa_span + "\n...\n"+ answer
        answer = answer[:max_chars]

    if len(answer) < 500 and len(scored_docs) > 1:
        for did, _, _ in scored_docs[1:3]:
            supplement = extract_answer_for_qa(question, did, corpus, max_chars - len(answer))
            if supplement and len(supplement) > 50:
                answer = answer + "\n\n"+ supplement
                if len(answer) >= max_chars:
                    break

    return answer[:max_chars].strip()


# Main Pipeline
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    zip_t1 = os.path.join(
        base_dir, "PublicTest",
        "LegalIR - Public Test-20260805T231927Z-1-001",
        "LegalIR - Public Test", "selected-contexts.zip"
    )
    fallback_dir = os.path.join(
        base_dir, "PublicTest",
        "LegalIR - Public Test-20260805T231927Z-1-001",
        "LegalIR - Public Test", "selected-contexts"
    )
    t1_test_path = os.path.join(
        base_dir, "PrivateTest", "Task1", "private-official.json"
    )
    t2_test_path = os.path.join(
        base_dir, "PrivateTest", "Task2", "private-official.json"
    )

    cache_corpus = os.path.join(base_dir, CACHE_CORPUS)
    cache_index = os.path.join(base_dir, CACHE_INDEX)
    device = "cuda"if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load / Build Corpus & Chunks
    if os.path.exists(cache_corpus):
        print("Load cached corpus")
        with open(cache_corpus, "rb") as f:
            corpus, doc_ids, doc_names, doc_raw_names, doc_citations, chunks = pickle.load(f)
        print(f"{len(corpus)} documents, {len(chunks)} chunks")
    else:
        print("Build corpus & chunks")
        corpus = load_corpus(zip_t1, fallback_dir)
        doc_ids = list(corpus.keys())
        doc_names = {did: clean_title(doc.get("name", "")) for did, doc in corpus.items()}
        doc_raw_names = {did: doc.get("name", "") for did, doc in corpus.items()}
        doc_citations = {
            did: extract_citations(doc_raw_names[did] + ""+ corpus[did]["passage"][:800])
            for did in doc_ids
        }
        chunks = []
        for did in doc_ids:
            chunks.extend(split_articles(did, doc_names[did], corpus[did]["passage"]))
        print(f"Built {len(corpus)} documents, {len(chunks)} chunks")
        with open(cache_corpus, "wb") as f:
            pickle.dump((corpus, doc_ids, doc_names, doc_raw_names, doc_citations, chunks), f)

    # 2. Build BM25 Indexes
    print("Build BM25 indexes")
    bm25_name = UltraFastBM25(
        [tokenize_vi(doc_names[did]) for did in doc_ids],
        min_idf=0.2
    )
    bm25_chunk = UltraFastBM25(
        [tokenize_vi(c["text"]) for c in chunks],
        min_idf=0.4
    )
    print("Build BM25 passage index")
    bm25_passage = UltraFastBM25(
        [tokenize_vi(corpus[did]["passage"][:5000]) for did in doc_ids],
        min_idf=0.3
    )

    # 3. Load ONNX Models
    print("No dense index (BM25 only phase 1)")
    embedder = None
    dense_index = None

    print("Load ONNX reranker (DirectML with Optimization)")
    ce_tokenizer = AutoTokenizer.from_pretrained("bge-reranker-onnx")
    
    sess_options = SessionOptions()
    sess_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    
    try:
        ce_model = ORTModelForSequenceClassification.from_pretrained(
            "bge-reranker-onnx", 
            provider="DmlExecutionProvider",
            session_options=sess_options
        )
        print("DirectML loaded with Graph Optimization!")
    except Exception as e:
        print(f"DirectML failed: {e}. Falling back to CPU")
        ce_model = ORTModelForSequenceClassification.from_pretrained(
            "bge-reranker-onnx", 
            provider="CPUExecutionProvider",
            session_options=sess_options
        )
    
    reranker = (ce_model, ce_tokenizer)

    print("Load QA Model (CPU Fast Pipeline)")
    qa_pipeline = None
    try:
        qa_pipeline = pipeline("question-answering", model=QA_MODEL, device=-1)
    except Exception as e:
        print(f"Failed to load QA model: {e}")
        try:
            qa_pipeline = pipeline("question-answering", model=QA_MODEL, tokenizer=AutoTokenizer.from_pretrained(QA_MODEL, use_fast=False), device=-1)
        except Exception as e2:
            print(f"[ERROR] QA model failed completely: {e2}")

    # 4. Task 1: LegalIR — Maximize Recall@5
    print("\n"+ "="* 50)
    print("Task 1: Legal IR (Maximize Recall@5)")
    print("="* 50)
    t1_queries = load_queries(t1_test_path)
    print(f"{len(t1_queries)} queries")

    t1_results = retrieve(
        t1_queries, corpus, chunks, doc_ids, doc_names, doc_citations,
        bm25_chunk, bm25_name, bm25_passage, embedder, dense_index, reranker,
        k_candidates=TASK1_K_CANDIDATES, use_safety_net=False, task_name="Task1"
    )

    t1_submission = {}
    for qid in t1_queries:
        if qid in t1_results and t1_results[qid]:
            scored = sorted(t1_results[qid], key=lambda x: x[1], reverse=True)
            top_docs = [d[0] for d in scored[:5]]
            if len(top_docs) < 5:
                for did in doc_ids:
                    if did not in top_docs:
                        top_docs.append(did)
                    if len(top_docs) >= 5:
                        break
            t1_submission[qid] = {"answer": top_docs[:5]}
        else:
            t1_submission[qid] = {"answer": doc_ids[:5]}

    t1_out = os.path.join(base_dir, "submission_task1.zip")
    with zipfile.ZipFile(t1_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", json.dumps(t1_submission, ensure_ascii=False, indent=2))
    print(f"Task 1 saved -> {t1_out}")

    # 5. Task 2: LegalQA — Maximize METEOR
    print("\n"+ "="* 50)
    print("Task 2: Legal QA (Maximize METEOR)")
    print("="* 50)
    t2_queries = load_queries(t2_test_path)
    print(f"{len(t2_queries)} queries")

    t2_results = retrieve(
        t2_queries, corpus, chunks, doc_ids, doc_names, doc_citations,
        bm25_chunk, bm25_name, bm25_passage, embedder, dense_index, reranker,
        k_candidates=TASK2_K_CANDIDATES, use_safety_net=False, task_name="Task2"
    )

    t2_submission = {}
    for qid in t2_queries:
        if qid in t2_results and t2_results[qid]:
            scored = sorted(t2_results[qid], key=lambda x: x[1], reverse=True)
            answer = extract_multi_doc_answer(
                t2_queries[qid], scored[:3], corpus, qa_pipeline,
                max_chars=TASK2_MAX_ANSWER_LEN
            )
            t2_submission[qid] = {"answer": answer}
        else:
            t2_submission[qid] = {"answer": ""}

    t2_out = os.path.join(base_dir, "submission_task2.zip")
    with zipfile.ZipFile(t2_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", json.dumps(t2_submission, ensure_ascii=False, indent=2))
    print(f"Task 2 saved -> {t2_out}")

    print("\n"+ "="* 50)
    print("DONE — Both submissions generated!")
    print("="* 50)


if __name__ == "__main__":
    main()