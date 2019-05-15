epochs=30
round=5
p=1.0

for alpha in 0.01 0.003 0.001 0.0003 0.0001 0.00003 0.00001 0.
do
    for i in $(seq 1 $round)
    do
        python train.py --alpha $alpha --suffix $i --epochs $epochs --p $p
    done
done
