```mermaid
%%{init: {'theme': 'default'}}%%
sequenceDiagram
    autonumber
    actor User
    participant CLI as "cli.main"
    participant SC as "SynthCurator"
    participant STD as "steps.standardize"
    participant FEAT as "steps.features"
    participant MET as "steps.metrics"
    participant MS as "steps.metric_sets"
    participant TOUR as "steps.tournament"
    participant MODEL as "Model (user code)"
    participant WGT as "steps.weights"
    participant CUR as "steps.curate"
    participant GRP as "core.groups"

    User->>CLI: synth-curator config.yaml [--train-production]
    CLI->>SC: SynthCurator.from_config(path)
    SC->>SC: _validate()
    Note right of SC: fails immediately if the model lacks train_and_evaluate, synthetic_dir or real_dir is missing or has no images, resolution is not positive, target_distribution does not sum to ~100, or metric_beta and scoring are both set, before any expensive work runs
    SC-->>CLI: instance (model, synthetic_dir, real_dir, resolution, metrics, curation config)

    alt --train-production flag given
        CLI->>SC: train_production()
        SC->>SC: check model has train_production()
        Note right of SC: fails immediately here, before curating, if the model only implements train_and_evaluate
    else default
        CLI->>SC: run()
    end

    Note over SC: full path only, no bring-your-own-weights yet - run() is one flat sequence, each step's output threaded explicitly into the next, no step pulls its own input. train_production() calls this same run() internally before doing anything extra

    SC->>STD: standardize() [step 1]
    Note right of STD: skip if real_standardized exists and its fingerprint (real_dir, resolution) matches, else recompute
    STD-->>SC: std_dir

    SC->>FEAT: extract_features(std_dir) [step 2]
    Note right of FEAT: skip if synthetic_dino.pt exists and the fingerprint (synthetic_dir, real_dir) matches, else recompute
    FEAT-->>SC: features_dir (dino, inception, probs, knn_map, real_labels)

    SC->>MET: compute_metrics(std_dir, features_dir) [step 3]
    Note right of MET: base metadata is metadata_csv if set, else scanned from synthetic_dir folders - a changed fingerprint clears the cached CSV first
    MET->>MET: paired.py / single_image.py / from_features.py calculators, each skipping columns already computed
    MET-->>SC: metadata_csv

    SC->>MS: curate_metric_sets(metadata_csv, features_dir) [step 4]
    opt distributional metrics configured and dist_group_by columns present
        MS->>GRP: compute_group_raw(df, features_dir, dist_group_by, dist_metrics)
        GRP-->>MS: raw FID / KID / SWD per generation group
    end
    MS->>MS: per instance metric: top-N per class
    MS->>MS: per distributional metric: best group by raw score, then random draw
    MS->>MS: random_control: fresh random per-class draw, one per seed
    MS-->>SC: metric_sets {metric_name: {seed: manifest.csv}}

    SC->>TOUR: run_tournament(metric_sets) [step 5]
    loop each metric_set x each seed
        TOUR->>TOUR: check cache {metric_set}_seed{seed}_predictions.csv
        alt predictions already cached and not force_retrain
            TOUR-->>TOUR: load predictions.csv, log the reuse
        else not cached, or force_retrain
            TOUR->>MODEL: train_and_evaluate(manifest_csv, seed)
            MODEL-->>TOUR: predictions (unique_image_id, true_label, predicted_label)
            TOUR->>TOUR: validate columns, cache to disk
        end
    end
    TOUR-->>SC: predictions {metric_set_name: DataFrame[..., seed]}

    SC->>WGT: derive_weights(predictions) [step 6]
    WGT->>WGT: per class: mean score per metric set across seeds, via scoring or F-metric_beta by default (2.0 = F2)
    WGT->>WGT: worst_anchored / random_anchored formula -> normalize to sum 1
    WGT->>WGT: optional dist_formula re-split within the distributional share only
    WGT-->>SC: weights {class: {metric: weight}}
    SC->>SC: write workdir/weights.json

    SC->>CUR: curate(metadata_csv, features_dir, weights, out_dir, curation params) [step 7]
    Note right of CUR: skip entirely if out_dir/seed_{seed}.csv already exists
    CUR-->>SC: manifest path

    alt --train-production flag given
        SC->>MODEL: train_production(manifest)
        MODEL-->>SC: artifact, a checkpoint path or a wrapped model or anything else, never inspected by the library
        SC-->>CLI: artifact
    else default
        SC-->>CLI: manifest path
    end
    CLI-->>User: print(result)
```
