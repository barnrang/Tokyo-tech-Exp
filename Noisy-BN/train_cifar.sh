epochs=50
round=5
p=1.0

run_training() {
    python train1.py --dataset cifar --alpha $1 --suffix $2 --epochs $3 --p $4 --gpu $5
}

for alpha in 0.1 0.03 0.01 0.003 0.001 0.0003 0.0001 0.00003 0.00001 0.
do
    for i in 0 1 2 3 4
    do
        run_training $alpha $i $epochs $p $i &
    done
    wait
done
