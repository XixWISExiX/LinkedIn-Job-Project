import pandas as pd
import numpy as np
import math


def ModifiedApriori(df, min_support, N):
    minsup_count = math.ceil(len(df) * min_support)

    k = 1
    mp = df.loc[:, df.columns != 'normalized_salary'].sum(numeric_only=True)
    Fk = {(i,): c for i, c in mp.items() if c >= minsup_count} # frequent filter
    freq_itemsets = {}
    if Fk and 1 <= N: freq_itemsets[1] = Fk.copy()

    while Fk and k < N:
        k += 1

        def candidate_gen(Fk):
            Ck = {}
            lexo_list = sorted(Fk.items())
            for i in range(len(lexo_list)):
                for j in range(i+1, len(lexo_list)):
                    key1, c1 = lexo_list[i]
                    key2, c2 = lexo_list[j]
                    if len(key1) < 2:
                        Ck[(key1[0], key2[0])] = 0
                    elif key1[:k-2] == key2[:k-2]:
                        tmp = list(key1)
                        tmp.append(key2[k-2:k-1][0])
                        Ck[tuple(tmp)] = 0
                    else:
                        # because list is lexicographically sorted, no later b will match a’s prefix
                        break
            return Ck

        Ck = candidate_gen(Fk)

        def candidate_prune(Ck):
            cols = [c for c in df.columns if c != 'normalized_salary']
            M = df[cols].to_numpy(dtype=np.bool_)
            idx = {c:i for i, c in enumerate(cols)}
            for key in list(Ck):
                cols = (key,) if isinstance(key, str) else tuple(key)
                try:
                    j = [idx[c] for c in cols]
                except KeyError:
                    Ck[key] = 0
                    continue
                Ck[key] = int(M[:, j].all(axis=1).sum())
            return Ck

        Ck = candidate_prune(Ck)

        Fk = {i: c for i, c in Ck.items() if c >= minsup_count} # frequent filter (the real pruning)
        freq_itemsets[k] = Fk

    return freq_itemsets



def RuleGeneration(Fk_union, df, min_support, N, K, sort_by):
    merged = {}
    seen = set()
    for d in Fk_union.values():
        merged.update(d)

    output_list = []

    bin_df = df.drop(columns='normalized_salary', errors='ignore').astype(bool)  # 1→True, 0→False

    # Optimized counting implementation for multiple columns
    #####################################
    skills = [c for c in df.columns if c != 'normalized_salary']
    B = df[skills].to_numpy(dtype=np.uint8)  # N×D (0/1)

    # pack rows into 64-bit words per column
    N = B.shape[0]
    W = (N + 63) // 64
    packed = np.zeros((W, B.shape[1]), dtype=np.uint64)
    for j in range(B.shape[1]):
        col = B[:, j].astype(np.uint64)
        # pack 64 bits at a time
        for w in range(W):
            start = w*64
            chunk = col[start:start+64]
            if chunk.size:
                packed[w, j] = np.packbits(chunk[::-1], bitorder='little').view(np.uint64)[0] \
                            if chunk.size==64 else int(''.join(map(str, chunk[::-1])), 2)

    col_idx = {c:i for i,c in enumerate(skills)}

    def count_union_bitset(cols):
        idxs = [col_idx[c] for c in cols]
        acc = np.uint64(~0)                      # all 1s
        for j in idxs:
            acc &= packed[:, j]
        return int(np.unpackbits(acc.view(np.uint8)).sum())
    #####################################

    lexo_list = sorted(merged.items())

    for i in range(len(lexo_list)):
        for j in range(i+1, len(lexo_list)):
            key1, c1 = lexo_list[i]
            key2, c2 = lexo_list[j]

            sup1 = c1/len(df)
            sup2 = c2/len(df)

            required = tuple(set(key1) | set(key2))

            # Efficient counting implementation
            count = count_union_bitset(required)

            sup_both = count/len(df)

            if sup_both < min_support: continue

            conf1 = sup_both/sup1
            conf2 = sup_both/sup2

            lift1 = conf1/sup2
            lift2 = conf2/sup1

            # avoid redundant rules
            A = frozenset(key1); B = frozenset(key2)
            if A & B: continue

            # don't have identical rules
            key = (A, B)
            if key in seen:
                continue
            seen.add(key)

            output_list.append((sup_both, conf1, lift1, key1, key2, ''))
            output_list.append((sup_both, conf2, lift2, key2, key1, ''))

    rules_df = pd.DataFrame(output_list, columns=['support', 'confidence', 'lift', 'antecedent', 'consequent', 'correlation'])

    rules_df[["support","confidence","lift"]] = rules_df[["support","confidence","lift"]].round(4)

    rules_df.loc[rules_df['lift'] > 1, 'correlation'] = 'positive'
    rules_df.loc[rules_df['lift'] < 1, 'correlation'] = 'negative'
    rules_df.loc[~((rules_df['lift'] > 1) | (rules_df['lift'] < 1)), 'correlation'] = 'independent'

    rules_df = rules_df.sort_values(sort_by, ascending=False)

    return rules_df[:K]



# min_support (min support probability)
# max_rule_size "N" (max tuple size per side) | e.g. N=3 ('a', 'b', 'c') -> ('d', 'e', 'f')
# topk "K" (topk rules sorted by confidence, returned)
def run_association_rule_mining(skills_df, min_support, max_rule_size, topk, sort_col):
    Fk_union = ModifiedApriori(skills_df, min_support=min_support, N=max_rule_size)

    return RuleGeneration(Fk_union, skills_df, min_support=min_support, N=max_rule_size, K=topk, sort_by=sort_col)




