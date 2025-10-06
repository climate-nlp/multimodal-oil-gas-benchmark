set -eu

SEED=42
PIXELS=262144  # 512*512

# (Full model) Few-shot + fewshot_entity_restriction + fewshot_embedding_search + transcript input
OUT_PREFIX=${OUTDIR}/predictions.${NUM_FEWSHOT}_shot
python -m src.predict \
    --apikey ${APIKEY} \
    --seed ${SEED} \
    --train_file ${TRAIN_FILE} \
    --test_file ${TEST_FILE} \
    --videodir ${VIDEO_DIR} \
    --transcriptdir ${TRANSCRIPT_DIR} \
    --prompt ${PROMPT} \
    --max_frames ${MAX_FRAMES} \
    --outfile ${OUT_PREFIX}.jsonl \
    --model ${MODEL} \
    --num_fewshot ${NUM_FEWSHOT} \
    --embedding_model ${EMBEDDING_MODEL} \
    --resized_pixels ${PIXELS} \
    --cachedir Cache/
python -m src.score -s ${OUT_PREFIX}.jsonl -g ${TEST_FILE} -o ${OUT_PREFIX}.csv


# Zero-shot
OUT_PREFIX=${OUTDIR}/predictions.zero_shot
python -m src.predict \
    --apikey ${APIKEY} \
    --seed ${SEED} \
    --train_file ${TRAIN_FILE} \
    --test_file ${TEST_FILE} \
    --videodir ${VIDEO_DIR} \
    --transcriptdir ${TRANSCRIPT_DIR} \
    --prompt ${PROMPT} \
    --max_frames ${MAX_FRAMES} \
    --outfile ${OUT_PREFIX}.jsonl \
    --model ${MODEL} \
    --num_fewshot 0 \
    --embedding_model ${EMBEDDING_MODEL} \
    --resized_pixels ${PIXELS} \
    --cachedir Cache/
python -m src.score -s ${OUT_PREFIX}.jsonl -g ${TEST_FILE} -o ${OUT_PREFIX}.csv
