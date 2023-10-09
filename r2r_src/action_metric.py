from utils import read_vocab, Tokenizer
import numpy as np
from tslearn.metrics import dtw_path_from_metric
from sklearn.metrics.pairwise import pairwise_distances


# from https://stackoverflow.com/questions/10459493/
def KMPSearch(pat, txt):
    results = []

    M = len(pat)
    N = len(txt)

    # create lps[] that will hold the longest split suffix
    # values for pattern
    lps = [0] * M
    j = 0  # index for pat[]

    # Preprocess the pattern (calculate lps[] array)
    computeLPSArray(pat, M, lps)

    i = 0  # index for txt[]
    while i < N:
        if pat[j] == txt[i]:
            i += 1
            j += 1

        if j == M:
            # print("Found pattern at index " + str(i-j))
            results.append(i - j)
            j = lps[j - 1]

        # mismatch after j matches
        elif i < N and pat[j] != txt[i]:
            # Do not match lps[0..lps[j-1]] characters,
            # they will match anyway
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return results


def computeLPSArray(pat, M, lps):
    len = 0  # length of the previous longest split suffix

    assert lps[0] == 0  # lps[0] is always 0
    i = 1

    # the loop calculates lps[i] for i = 1 to M-1
    while i < M:
        if pat[i] == pat[len]:
            len += 1
            lps[i] = len
            i += 1
        else:
            # This is tricky. Consider the example.
            # AAACAAAA and i = 7. The idea is similar
            # to search step.
            if len != 0:
                len = lps[len - 1]

                # Also, note that we do not increment i here
            else:
                lps[i] = 0
                i += 1


# a = [2, 3, 5, 2, 5, 6, 7, 2, 5, 6]
# b = [2, 5, 6]
# KMPSearch(b, a)


def setup():
    vocab = read_vocab('../tasks/R2R/data/train_vocab.txt')
    tok = Tokenizer(vocab=vocab, encoding_length=80)

    # directional
    directional = dict()  # left 0, right 1, around 2
    for ins in ['turn right', 'turn to the right', 'make a right', 'veer right', 'take a right']:
        directional[ins] = 0
    for ins in ['turn left', 'turn to the left', 'make a left', 'veer left', 'take a left']:
        directional[ins] = 1
    for ins in ['turn around', 'turn 180 degrees', 'make a 180 degree turn', 'veer around']:
        directional[ins] = 2

    all_directional = []
    all_directional_type = []

    for k, v in directional.items():
        all_directional.append([tok.word_to_index[word] for word in tok.split_sentence(k)])
        all_directional_type.append(v)

    return all_directional, all_directional_type, tok


def parse_action(sentence, all_directional, all_directional_type):
    act_positions = []
    act_types = []

    for i, d_phrase in enumerate(all_directional):
        # print(d_phrase, KMPSearch(d_phrase, ref))
        matching_results = KMPSearch(d_phrase, sentence)
        if len(matching_results) > 0:
            act_positions.extend(matching_results)
            act_types.extend([all_directional_type[i] for _ in range(len(matching_results))])

    act_positions = np.asarray(act_positions)
    act_types = np.asarray(act_types)

    argsort_ = np.argsort(act_positions)

    # act_positions = act_positions[argsort_]
    act_types = act_types[argsort_]

    return act_types


def action_metric(cand, ref, all_directional, all_directional_type):
    '''
        cand, ref should be list of word indices
    '''

    cand_action_types = parse_action(cand, all_directional, all_directional_type)
    ref_action_types = parse_action(ref, all_directional, all_directional_type)

    print('cand_action_types', cand_action_types)
    print('ref_action_types', ref_action_types)

    lcs = LCS(cand_action_types, ref_action_types)

    precision = float(lcs) / len(cand_action_types)
    recall = float(lcs) / len(ref_action_types)

    f1_score = 2 * precision * recall / (precision + recall)

    print(precision, recall)

    return f1_score


def LCS(seq_a, seq_b):
    """
        return the length of the longest common subsequence of seq_a and seq_b
    """

    dp = np.zeros((len(seq_a) + 1, len(seq_b) + 1))

    for i, x in enumerate(seq_a):
        for j, y in enumerate(seq_b):
            if x == y:
                dp[i + 1, j + 1] = dp[i, j] + 1
            else:
                dp[i + 1, j + 1] = max(dp[i + 1, j], dp[i, j + 1])

    return dp[len(seq_a), len(seq_b)]


def main():
    all_directional, all_directional_type, tok = setup()
    # print(all_directional)

    # cand = "turn right, make a right, turn left"
    cand = "turn around, make a right, turn left, turn left"

    ref = "You should leave the area with the desk and turn right to get to the sitting room. Once you enter the room make a right and go around the couch, then make a right to go through the doorway. Now make a right go to the end of hall and make a left so you can wait at the stairs. "

    cand = [tok.word_to_index[word] for word in tok.split_sentence(cand)]
    ref = [tok.word_to_index[word] for word in tok.split_sentence(ref)]
    print(ref)

    # cand = "Turn around and enter the hallway.  Walk past the stairs and stand near the bathroom door. "

    score = action_metric(cand, ref, all_directional, all_directional_type)
    print(score)


if __name__ == '__main__':
    main()

# def DTW(seq_a, seq_b, band_width=None):
#     """
#     DTW is used to find the optimal alignment path;
#     seq_a: candidate
#     seq_b: reference
#     """
#     seq_a = np.asarray(seq_a).reshape(-1, 1)
#     seq_b = np.asarray(seq_b).reshape(-1, 1)
#
#     # dist for different types is 1, otherwise 0
#     dist_matrix = pairwise_distances(seq_a, seq_b, metric="euclidean")
#     dist_matrix = (dist_matrix > 0).astype(float) * 1
#
#     if band_width is None:
#         align_pairs, dist = dtw_path_from_metric(dist_matrix,
#                                                  metric="precomputed")
#     else:
#         align_pairs, dist = dtw_path_from_metric(dist_matrix,
#                                                  metric="precomputed",
#                                                  sakoe_chiba_radius=band_width)
#
#     # filter the alignment such that each action in the shorter sequence is only aligned to one action in the other sequence
#     alignment = np.ones((seq_a.shape[0], seq_b.shape[0])) * 999
#
#     for i in range(len(align_pairs)):
#         alignment[align_pairs[i][0], align_pairs[i][1]] = 0
#
#     dist_matrix += alignment  # distances of non-aligned pairs in DTW are set to 999+
#
#     print(dist_matrix)
#
#     if seq_a.shape[0] < seq_b.shape[0]:
#         filtered_dist = np.min(dist_matrix, axis=1)
#         assert filtered_dist.shape[0] == seq_a.shape[0]
#         filtered_dist = filtered_dist.sum()
#     else:
#         filtered_dist = np.min(dist_matrix, axis=0)
#         assert filtered_dist.shape[0] == seq_b.shape[0]
#         filtered_dist = filtered_dist.sum()
#
#     # calculate precision and recall
#     #   seq_a: candidate
#     #   seq_b: reference
#     #   filtered_dist: the number of mismatches under the optimal one-to-one alignment
#
#     precision = float(np.minimum(seq_a.shape[0], seq_b.shape[0]) - filtered_dist) / seq_a.shape[0]
#     recall = float(np.minimum(seq_a.shape[0], seq_b.shape[0]) - filtered_dist) / seq_b.shape[0]
#
#     return precision, recall
