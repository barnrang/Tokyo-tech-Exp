epochs=50
round=5
p=1.0
train_size=5000

run_training() {
    python train.py --dataset cifar --alpha $3 --suffix $2 --epochs $3 --p $4 --gpu $5 --train_size $train_size --small
}

> err.txt # Clear err.txt

for alpha in 0.03 0.01 0.003 0.001 0.0003 0.0001 0.00003 0.00001 0.0
do
    for i in 1 2 3 4 5
    do
        run_training $alpha $i$train_size $epochs $p $i 2>> err.txt &
    done
    wait
done
