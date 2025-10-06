set -eu

SEED=42
PIXELS=65536  # 256*256

# Few-shot + fewshot_embedding_search + transcript input
for NUM_FEWSHOT in 8 4
do
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
done
