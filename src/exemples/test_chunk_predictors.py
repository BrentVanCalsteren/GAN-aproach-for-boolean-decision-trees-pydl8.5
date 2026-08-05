import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pydl85 import DL85Classifier

import CONFIG
from data.data_obj.sampels import Samples
from src.usePydl.predictors.local_greedy_predictors import build_tree_iteratively

DO_CLASSIFICATION = True


def test_chunk_data_generation(dataset: str = 'bank', data_type: str = 'tabular', samples_per_chunk: int = 100):
    print(f"=== Starting Multi-Chunk Synthetic Data Generation for '{dataset}' ===")

    # 1. Initialize Samples object (automatically scans chunks and computes global stats)
    sample_obj = Samples(dataset=dataset, data_type=data_type)
    n_chunks = sample_obj.loader.n_chunks
    print(f"Dataset Loaded: '{dataset}' ({data_type}) with {n_chunks} chunk(s).")
    print(f"Global Feature Bounds (Min): {sample_obj.chunk_info.feats_min_vals}")
    print(f"Global Feature Bounds (Max): {sample_obj.chunk_info.feats_max_vals}")
    if sample_obj.chunk_info.feature_importance is not None:
        print(f"Global Feature Importance Ratio: {sample_obj.chunk_info.feature_importance[:5]}...")

    all_original_splits = []
    all_original_labels = []
    all_generated_splits = []
    all_generated_labels = []

    # 2. Process each chunk iteratively
    for chunk_id in range(n_chunks):
        print(f"\n------------------ Processing Chunk {chunk_id + 1}/{n_chunks} ------------------")
        sample_obj.load_chunk(chunk_id)
        raw_samples = sample_obj.samples
        chunk_labels = sample_obj.labels.flatten()

        print(f"Chunk {chunk_id} loaded. Samples shape: {raw_samples.shape}, Labels: {len(chunk_labels)}")

        # Save original preprocessed output
        sample_obj.save_output(samples=raw_samples, llables=None, output_name=f'original_chunk_{chunk_id}')

        # Create boolean splits for current chunk
        feature_data = sample_obj.current_feat_hist
        feature_data.creat_splits(total_num_splits=CONFIG.MAX_BOOL_SPLITS)
        chunk_splits = feature_data.get_splits()

        # Validate boolean mapping
        mapped_splits = feature_data.splits_obj.map_samples_to_splits(samples=raw_samples)
        splits_valid = np.equal(chunk_splits, mapped_splits).all()
        print(f"Boolean splits conversion valid: {splits_valid}")

        # Build DL8.5 greedy decision tree on chunk
        print(f"Building iterative greedy decision tree on chunk {chunk_id}...")
        ensemble_pred = build_tree_iteratively(feature_data)

        # Evaluate leaf purity on chunk samples
        complete_tree = feature_data.get_complete_tree()
        if complete_tree is not None:
            avg_purity, leaf_info = complete_tree.tree_label_distr_each_leaf(samples=raw_samples, labels=chunk_labels)
            print(f"Chunk {chunk_id} Leaf Purity: {avg_purity:.4f} across {len(leaf_info)} leaves.")

        # Generate synthetic samples based on chunk decision tree structure
        print(f"Generating {samples_per_chunk} synthetic samples for chunk {chunk_id}...")
        samples_gen = ensemble_pred.gen_new_data_based_tree_structure(n=samples_per_chunk, conf=0.0)

        # Match generated samples to closest label
        labels_gen = sample_obj.get_best_matching_label(samples=samples_gen, chunk_id=chunk_id)

        # Save generated output chunk
        sample_obj.save_output(samples=samples_gen, llables=labels_gen, output_name=f'generated_chunk_{chunk_id}')

        # Map generated samples to boolean splits
        gen_splits = feature_data.splits_obj.map_samples_to_splits(samples_gen)

        all_original_splits.append(chunk_splits)
        all_original_labels.append(chunk_labels)
        all_generated_splits.append(gen_splits)
        all_generated_labels.append(labels_gen)

        # Evaluate single-chunk classification
        if DO_CLASSIFICATION:
            print(f"\n--- Classification Evaluation for Chunk {chunk_id} ---")
            train_x, test_x, train_y, test_y = train_test(chunk_splits, chunk_labels, test_size=0.2)
            print("[Original Chunk Data Alone]:")
            classify(train_x, test_x, train_y, test_y)

            print("[Generated Chunk Data Alone]:")
            classify(gen_splits, test_x, labels_gen, test_y)

            combined_x = np.vstack((train_x, gen_splits))
            combined_y = np.hstack((train_y, labels_gen))
            print("[Combined Chunk Data (Original + Generated)]:")
            classify(combined_x, test_x, combined_y, test_y)

    # 3. Combined Multi-Chunk Evaluation
    if DO_CLASSIFICATION and n_chunks > 1:
        print("\n================ Multi-Chunk Aggregated Evaluation ================")
        full_orig_x = np.vstack(all_original_splits)
        full_orig_y = np.hstack(all_original_labels)
        full_gen_x = np.vstack(all_generated_splits)
        full_gen_y = np.hstack(all_generated_labels)

        train_x, test_x, train_y, test_y = train_test(full_orig_x, full_orig_y, test_size=0.2)
        print("[All Chunks Original Data]:")
        classify(train_x, test_x, train_y, test_y)

        print("[All Chunks Generated Data]:")
        classify(full_gen_x, test_x, full_gen_y, test_y)

        combined_full_x = np.vstack((train_x, full_gen_x))
        combined_full_y = np.hstack((train_y, full_gen_y))
        print("[All Chunks Combined Data]:")
        classify(combined_full_x, test_x, combined_full_y, test_y)

    print("\n=== Multi-Chunk Data Generation Test Complete! ===")


def train_test(splits, samples, test_size=0.2):
    return train_test_split(splits, samples, test_size=test_size, random_state=random.randint(1, 100))


def classify(x_train, x_test, y_train, y_test):
    if DO_CLASSIFICATION:
        depth = 1
        uniques = np.unique(y_train)
        while 2 ** (depth - 1) < len(uniques):
            depth += 1
        clasfi = create_classifier_default(x_bin=x_train, y=y_train, max_depth=depth, min_sup=1, time=100)
        y_pred_test = clasfi.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred_test)
        print(f"Accuracy: {accuracy:.4f}")


def create_classifier_default(x_bin, y, max_depth=3, min_sup=2, time=100):
    clasfi = DL85Classifier(max_depth=max_depth, min_sup=min_sup, time_limit=time)
    clasfi.fit(x_bin, y)
    return clasfi


if __name__ == "__main__":
    test_chunk_data_generation()
