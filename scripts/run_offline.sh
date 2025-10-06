set -eu

SEED=42
PIXELS=65536  # 256*256

# (Full model) Few-shot + fewshot_entity_restriction + fewshot_embedding_search + transcript input
OUT_PREFIX=${OUTDIR}/predictions.${NUM_FEWSHOT}_shot
python -m src.predict \
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


# (Ablations below)

# Few-shot w/o fewshot_entity_restriction
OUT_PREFIX=${OUTDIR}/predictions.no_fewshot_entity_restriction.${NUM_FEWSHOT}_shot
python -m src.predict \
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
    --cachedir Cache/ \
    --no_fewshot_entity_restriction
python -m src.score -s ${OUT_PREFIX}.jsonl -g ${TEST_FILE} -o ${OUT_PREFIX}.csv


# Few-shot w/o fewshot_embedding_search
OUT_PREFIX=${OUTDIR}/predictions.no_fewshot_embedding_search.${NUM_FEWSHOT}_shot
python -m src.predict \
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
    --cachedir Cache/ \
    --no_fewshot_embedding_search
python -m src.score -s ${OUT_PREFIX}.jsonl -g ${TEST_FILE} -o ${OUT_PREFIX}.csv


# Few-shot w/o transcript
OUT_PREFIX=${OUTDIR}/predictions.no_transcript.${NUM_FEWSHOT}_shot
python -m src.predict \
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
    --cachedir Cache/ \
    --no_transcript
python -m src.score -s ${OUT_PREFIX}.jsonl -g ${TEST_FILE} -o ${OUT_PREFIX}.csv


# Zero-shot
OUT_PREFIX=${OUTDIR}/predictions.zero_shot
python -m src.predict \
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

