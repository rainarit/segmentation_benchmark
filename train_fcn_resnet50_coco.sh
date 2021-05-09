

LEARNING_RATE=1e-6
BATCH_SIZE=2
WORKERS=2


echo "Running Model"

python core/scripts/train.py \
 --lr=${LEARNING_RATE} \
 --batch-size=${BATCH_SIZE} \
 --workers=${WORKERS}