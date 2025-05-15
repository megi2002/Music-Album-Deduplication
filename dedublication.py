import xml.dom.minidom as minidom
import jellyfish
from sklearn.metrics import precision_recall_fscore_support


# ---------- Step 1: Parse discs using DOM ----------
def parse_discs_dom(file_path):
    dom = minidom.parse(file_path)
    discs = {}
    for disc in dom.getElementsByTagName("disc"):
        disc_id = disc.getElementsByTagName("id")[0].firstChild.nodeValue if disc.getElementsByTagName("id") else None
        if not disc_id:
            cid_nodes = disc.getElementsByTagName("cid")
            disc_id = cid_nodes[0].firstChild.nodeValue if cid_nodes else None
        if not disc_id:
            continue

        artist_nodes = disc.getElementsByTagName("artist")
        title_nodes = disc.getElementsByTagName("dtitle")
        track_nodes = disc.getElementsByTagName("title")

        artist = artist_nodes[0].firstChild.nodeValue.strip() if artist_nodes and artist_nodes[0].firstChild else ""
        title = title_nodes[0].firstChild.nodeValue.strip() if title_nodes and title_nodes[0].firstChild else ""
        tracks = [node.firstChild.nodeValue.strip() for node in track_nodes if node.firstChild]

        discs[disc_id] = {
            "artist": artist,
            "title": title,
            "tracks": tracks
        }
    return discs


# ---------- Step 2: Parse ground truth using DOM ----------
def parse_ground_truth_dom(file_path):
    dom = minidom.parse(file_path)
    pairs = set()
    for pair in dom.getElementsByTagName("pair"):
        ids = []
        for disc in pair.getElementsByTagName("disc"):
            id_node = disc.getElementsByTagName("id")
            cid_node = disc.getElementsByTagName("cid")
            disc_id = id_node[0].firstChild.nodeValue if id_node else (
                cid_node[0].firstChild.nodeValue if cid_node else None)
            if disc_id:
                ids.append(disc_id)
        if len(ids) == 2:
            pairs.add(tuple(sorted(ids)))
    return pairs


# ---------- Step 3: Similarity Functions ----------
def string_similarity(s1, s2):
    return jellyfish.jaro_winkler_similarity(s1.lower(), s2.lower())


def jaccard_similarity(list1, list2):
    set1, set2 = set(list1), set(list2)
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def compute_disc_similarity(d1, d2):
    artist_sim = string_similarity(d1["artist"], d2["artist"])
    title_sim = string_similarity(d1["title"], d2["title"])
    track_sim = jaccard_similarity(d1["tracks"], d2["tracks"])
    return 0.4 * artist_sim + 0.4 * title_sim + 0.2 * track_sim


# ---------- Step 4: Match Discs ----------
def match_discs(discs, threshold):
    matched_pairs = set()
    disc_ids = list(discs.keys())
    for i in range(len(disc_ids)):
        for j in range(i + 1, len(disc_ids)):
            id1, id2 = disc_ids[i], disc_ids[j]
            sim = compute_disc_similarity(discs[id1], discs[id2])
            if sim >= threshold:
                matched_pairs.add(tuple(sorted((id1, id2))))
    return matched_pairs


# ---------- Step 5: Evaluate Matches ----------
def evaluate(predicted, ground_truth):
    all_pairs = predicted | ground_truth
    y_true = [1 if pair in ground_truth else 0 for pair in all_pairs]
    y_pred = [1 if pair in predicted else 0 for pair in all_pairs]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    accuracy = correct / len(y_true) if y_true else 0.0

    return precision, recall, f1, accuracy


# ---------- Step 6: Run pipeline ----------
if __name__ == "__main__":
    # Parse discs and ground truth files
    discs = parse_discs_dom("cddb_discs.xml")
    ground_truth = parse_ground_truth_dom("cddb_9763_dups.xml")

    threshold = 0.8  # You can tweak this value as needed
    predicted_matches = match_discs(discs, threshold)

    # Save predicted matches to a file
    with open("matched_pairs.txt", "w") as f:
        for id1, id2 in sorted(predicted_matches):
            f.write(f"{id1},{id2}\n")

    # Evaluate predicted matches against the ground truth
    precision, recall, f1, accuracy = evaluate(predicted_matches, ground_truth)

    print(f"\nEvaluation with threshold {threshold}:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"\nPredicted matches: {len(predicted_matches)}")
    print(f"Ground truth pairs: {len(ground_truth)}")
